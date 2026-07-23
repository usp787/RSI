"""Shared configuration, hashing, provenance, and artifact helpers.

All callers are cluster entry points. Importing this module performs no network,
dataset, model, or GPU work.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "vllm",
    "math-verify",
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if re.search(r"\$\{[^}]+\}", expanded):
            raise ValueError(f"Unresolved environment variable in config value: {value}")
        return expanded
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_experiment(config_path: str | Path, experiment_name: str) -> dict[str, Any]:
    """Resolve defaults, dataset, model, and an experiment override."""
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError(f"Unsupported config version in {path}: {raw.get('version')}")
    try:
        experiment = raw["experiments"][experiment_name]
        dataset_key = experiment["dataset_key"]
        model_key = experiment["model_key"]
        dataset = deep_merge(raw.get("data_defaults", {}), raw["datasets"][dataset_key])
        model = raw["models"][model_key]
    except KeyError as exc:
        raise KeyError(f"Invalid experiment {experiment_name!r}: missing {exc}") from exc

    resolved = deep_merge(raw["defaults"], experiment)
    resolved.update(
        {
            "config_version": raw["version"],
            "config_path": str(path),
            "experiment_name": experiment_name,
            "dataset_key": dataset_key,
            "model_key": model_key,
            "dataset": dataset,
            "model": copy.deepcopy(model),
            "storage": copy.deepcopy(raw["storage"]),
        }
    )
    return _expand_env(resolved)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(*parts: Any, modulo: int = 2**31 - 1) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % modulo


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, newline="\n"
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(target)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_immutable_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    """Write JSONL once; an existing byte-different artifact is an error."""
    target = Path(path)
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if existing != payload:
            raise FileExistsError(f"Refusing to overwrite different immutable artifact: {target}")
    else:
        atomic_write_text(target, payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def manifest_dir(config: dict[str, Any]) -> Path:
    return Path(config["storage"]["data_root"]) / "manifests" / config["dataset_key"]


def experiment_dir(config: dict[str, Any]) -> Path:
    return Path(config["storage"]["artifact_root"]) / config["experiment_name"]


def checkpoint_dir(config: dict[str, Any], model_round: int) -> Path:
    return (
        Path(config["storage"]["checkpoint_root"])
        / config["experiment_name"]
        / f"m{model_round}"
    )


def generation_dir(config: dict[str, Any], phase: str, model_round: int) -> Path:
    return experiment_dir(config) / "raw" / phase / f"m{model_round}"


def generation_shard_path(
    config: dict[str, Any], phase: str, model_round: int, shard_index: int, shard_count: int
) -> Path:
    return generation_dir(config, phase, model_round) / (
        f"shard-{shard_index:03d}-of-{shard_count:03d}.jsonl"
    )


def select_deterministic_subset(
    rows: list[dict[str, Any]], limit: int | None, seed: int, namespace: str
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            stable_int(namespace, seed, row["problem_id"]),
            row["problem_id"],
        ),
    )
    if limit is None:
        return ordered
    if limit <= 0:
        raise ValueError(f"Subset limit must be positive or null, got {limit}")
    return ordered[: min(limit, len(ordered))]


def current_code_commit() -> str:
    declared = os.environ.get("CODE_COMMIT")
    if declared:
        return declared.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "code_commit": current_code_commit(),
        "config_sha256": canonical_sha256(config),
        "experiment": config["experiment_name"],
        "dataset": config["dataset_key"],
        "model": config["model_key"],
        "packages": package_versions(),
    }
