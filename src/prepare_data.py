"""Create immutable train/validation/evaluation manifests on cluster storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    load_experiment,
    manifest_dir,
    stable_int,
    text_sha256,
    write_immutable_jsonl,
)
from verify_math import extract_last_boxed, normalize_gsm8k

MANIFEST_SCHEMA_VERSION = 1


def _load_source(source: dict[str, Any]):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source["hub_id"],
        "split": source["split"],
        "revision": source["revision"],
    }
    if source.get("subset") is not None:
        kwargs["name"] = source["subset"]
    return load_dataset(**kwargs)


def _gsm8k_answer(solution: str) -> str | None:
    candidate = solution.rsplit("####", maxsplit=1)[-1].strip() if "####" in solution else solution
    return candidate if normalize_gsm8k(candidate) is not None else None


def _normalise_rows(
    dataset_key: str,
    dataset_config: dict[str, Any],
    source_config: dict[str, Any],
    source_split_name: str,
    source_rows: Any,
) -> tuple[list[dict[str, Any]], int]:
    fields = dataset_config["fields"]
    normalised: list[dict[str, Any]] = []
    skipped = 0
    for index, example in enumerate(source_rows):
        problem = str(example[fields["problem"]]).strip()
        solution = str(example.get(fields["solution"], "")).strip()
        if dataset_key == "gsm8k":
            answer = _gsm8k_answer(solution)
        else:
            answer_field = fields.get("answer")
            answer = str(example.get(answer_field, "")).strip() if answer_field else ""
            answer = answer or extract_last_boxed(solution)
        if not problem or not answer or not solution:
            skipped += 1
            continue
        content_hash = text_sha256(" ".join(problem.split()))
        row = {
            "problem_id": f"{dataset_key}:{source_split_name}:{index:06d}",
            "dataset": dataset_key,
            "source_hub_id": source_config["hub_id"],
            "source_revision": source_config["revision"],
            "source_split": source_config["split"],
            "source_index": index,
            "problem": problem,
            "answer": str(answer).strip(),
            "reference_solution": solution,
            "content_sha256": content_hash,
        }
        if dataset_key == "math":
            subject_field = fields.get("subject")
            fallback = fields.get("fallback_subject")
            row["subject"] = str(
                example.get(subject_field) or example.get(fallback) or "unknown"
            )
            row["level"] = str(example.get(fields.get("level"), "unknown"))
        normalised.append(row)
    return normalised, skipped


def _assert_no_eval_leakage(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    train_hashes = {row["content_sha256"] for row in train_rows}
    overlap = sorted(train_hashes.intersection(row["content_sha256"] for row in eval_rows))
    if overlap:
        raise ValueError(f"Train/eval content overlap detected for {len(overlap)} problem(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/restem.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    config = load_experiment(args.config, args.experiment)
    dataset_config = config["dataset"]
    output_dir = manifest_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_contract = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_key": config["dataset_key"],
        "dataset": dataset_config,
    }
    contract_hash = canonical_sha256(data_contract)
    success_path = output_dir / "_SUCCESS.json"
    if success_path.exists():
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success.get("data_contract_sha256") != contract_hash:
            raise FileExistsError(
                f"Existing manifest contract differs at {output_dir}; preserve it and choose a new root"
            )
        for item in success.get("files", {}).values():
            artifact = Path(item["path"])
            if not artifact.exists() or file_sha256(artifact) != item["sha256"]:
                raise FileNotFoundError(f"Manifest success marker references missing/changed file: {artifact}")
        print(f"[data] immutable manifests already complete: {output_dir}")
        return

    train_source = dataset_config["train_source"]
    eval_source = dataset_config["eval_source"]
    print(f"[data] loading {train_source['hub_id']}@{train_source['revision']}")
    train_dataset = _load_source(train_source)
    print(f"[data] loading {eval_source['hub_id']}@{eval_source['revision']}")
    eval_dataset = _load_source(eval_source)

    train_rows, train_skipped = _normalise_rows(
        config["dataset_key"], dataset_config, train_source, "train", train_dataset
    )
    eval_rows, eval_skipped = _normalise_rows(
        config["dataset_key"], dataset_config, eval_source, "eval", eval_dataset
    )
    _assert_no_eval_leakage(train_rows, eval_rows)

    validation_count = max(1, round(len(train_rows) * dataset_config["validation_fraction"]))
    ranked = sorted(
        train_rows,
        key=lambda row: (
            stable_int("validation", dataset_config["split_seed"], row["problem_id"]),
            row["problem_id"],
        ),
    )
    validation_rows = sorted(ranked[:validation_count], key=lambda row: row["problem_id"])
    train_pool_rows = sorted(ranked[validation_count:], key=lambda row: row["problem_id"])
    eval_rows = sorted(eval_rows, key=lambda row: row["problem_id"])

    files: dict[str, dict[str, Any]] = {}
    for name, rows in (
        ("train", train_pool_rows),
        ("validation", validation_rows),
        ("eval", eval_rows),
    ):
        path = output_dir / f"{name}.jsonl"
        digest = write_immutable_jsonl(path, rows)
        files[name] = {"path": str(path), "rows": len(rows), "sha256": digest}

    metadata = {
        "data_contract_sha256": contract_hash,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_key": config["dataset_key"],
        "train_source": train_source,
        "train_source_fingerprint": getattr(train_dataset, "_fingerprint", "unknown"),
        "eval_source": eval_source,
        "eval_source_fingerprint": getattr(eval_dataset, "_fingerprint", "unknown"),
        "split_seed": dataset_config["split_seed"],
        "validation_fraction": dataset_config["validation_fraction"],
        "skipped_rows": {"train": train_skipped, "eval": eval_skipped},
        "files": files,
    }
    atomic_write_json(output_dir / "manifest.json", metadata)
    metadata["manifest_sha256"] = file_sha256(output_dir / "manifest.json")
    atomic_write_json(success_path, metadata)
    print(
        f"[data] wrote train={len(train_pool_rows)} validation={len(validation_rows)} "
        f"eval={len(eval_rows)} -> {output_dir}"
    )


if __name__ == "__main__":
    main()
