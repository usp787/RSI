"""Verify raw training generations and build a capped, traceable SFT corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    atomic_write_json,
    canonical_sha256,
    experiment_dir,
    file_sha256,
    generation_dir,
    iter_jsonl,
    load_experiment,
    provenance,
    stable_int,
    write_immutable_jsonl,
)
from verify_math import verify_completion


def _discover_complete_shards(directory: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    success_files = sorted(directory.glob("shard-*.success.json"))
    if not success_files:
        raise FileNotFoundError(f"No completed generation shards in {directory}")
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in success_files]
    shard_counts = {item["shard_count"] for item in metadata}
    if len(shard_counts) != 1:
        raise ValueError(f"Mixed shard counts in {directory}: {shard_counts}")
    shard_count = shard_counts.pop()
    indices = {item["shard_index"] for item in metadata}
    if indices != set(range(shard_count)):
        raise ValueError(f"Incomplete shards in {directory}: have {sorted(indices)}, need 0..{shard_count - 1}")
    raw_paths = [path.with_name(path.name.replace(".success.json", ".jsonl")) for path in success_files]
    return raw_paths, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/restem.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--round", type=int, required=True, dest="model_round")
    args = parser.parse_args()

    config = load_experiment(args.config, args.experiment)
    if not 0 <= args.model_round < config["rounds"]:
        raise ValueError(f"Training generation round must be in [0, {config['rounds'] - 1}]")
    raw_dir = generation_dir(config, "train", args.model_round)
    raw_paths, raw_metadata = _discover_complete_shards(raw_dir)
    output_dir = experiment_dir(config) / "sft" / f"round-{args.model_round}"
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_contract = {
        **provenance(config),
        "model_round": args.model_round,
        "filter": config["filter"],
        "raw_contracts": sorted(item["generation_contract_sha256"] for item in raw_metadata),
        "raw_files": {str(path): file_sha256(path) for path in raw_paths},
    }
    contract_hash = canonical_sha256(filter_contract)
    success_path = output_dir / "_SUCCESS.json"
    if success_path.exists():
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success.get("filter_contract_sha256") != contract_hash:
            raise FileExistsError(f"Filter contract changed for existing output: {output_dir}")
        print(f"[filter] already complete: {output_dir}")
        return

    samples: dict[str, dict[str, Any]] = {}
    per_problem_raw: Counter[str] = Counter()
    for raw_path in raw_paths:
        for row in iter_jsonl(raw_path):
            sample_id = row["sample_id"]
            if sample_id in samples:
                raise ValueError(f"Duplicate sample_id across raw shards: {sample_id}")
            samples[sample_id] = row
            per_problem_raw[row["problem_id"]] += 1
    expected_per_problem = int(config["generation"]["train"]["samples_per_problem"])
    bad_counts = {
        problem_id: count
        for problem_id, count in per_problem_raw.items()
        if count != expected_per_problem
    }
    if bad_counts:
        preview = list(sorted(bad_counts.items()))[:5]
        raise ValueError(f"Raw sample count mismatch for {len(bad_counts)} problems: {preview}")

    verified_rows: list[dict[str, Any]] = []
    correct_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    parse_count = 0
    correct_count = 0
    duplicate_count = 0
    seen_completions: dict[str, set[str]] = defaultdict(set)
    for sample_id in sorted(samples):
        row = samples[sample_id]
        verification = verify_completion(
            config["dataset"]["verifier"],
            row["completion"],
            row["answer"],
            int(config["filter"]["verifier_timeout_seconds"]),
        )
        parse_count += int(verification["parsed"])
        correct_count += int(verification["correct"])
        if verification["failure_reason"]:
            failure_counts[verification["failure_reason"]] += 1
        completion_hash = row["completion_sha256"]
        is_duplicate = completion_hash in seen_completions[row["problem_id"]]
        seen_completions[row["problem_id"]].add(completion_hash)
        duplicate_count += int(is_duplicate)
        verified = {
            "sample_id": sample_id,
            "problem_id": row["problem_id"],
            "completion_sha256": completion_hash,
            "is_duplicate_within_problem": is_duplicate,
            **verification,
        }
        verified_rows.append(verified)
        if verification["correct"] and not (
            config["filter"]["deduplicate_completions"] and is_duplicate
        ):
            correct_by_problem[row["problem_id"]].append(row)

    cap = int(config["filter"]["max_accepted_per_problem"])
    accepted: list[dict[str, Any]] = []
    for problem_id in sorted(correct_by_problem):
        candidates = sorted(
            correct_by_problem[problem_id],
            key=lambda row: (
                stable_int("accept", config["seed"], args.model_round, row["sample_id"]),
                row["sample_id"],
            ),
        )[:cap]
        for row in candidates:
            accepted.append(
                {
                    "sample_id": row["sample_id"],
                    "problem_id": row["problem_id"],
                    "source": "direct",
                    "generator_round": args.model_round,
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "completion_sha256": row["completion_sha256"],
                }
            )

    verification_path = output_dir / "verification.jsonl"
    accepted_path = output_dir / "accepted.jsonl"
    verification_sha = write_immutable_jsonl(verification_path, verified_rows)
    accepted_sha = write_immutable_jsonl(accepted_path, accepted)
    total = len(samples)
    summary = {
        **filter_contract,
        "filter_contract_sha256": contract_hash,
        "total_samples": total,
        "parsed_samples": parse_count,
        "correct_samples": correct_count,
        "retained_samples": len(accepted),
        "duplicate_samples": duplicate_count,
        "total_problems": len(per_problem_raw),
        "problems_with_correct": len(correct_by_problem),
        "problem_coverage": len(correct_by_problem) / len(per_problem_raw),
        "parse_rate": parse_count / total,
        "correct_rate": correct_count / total,
        "failure_counts": dict(sorted(failure_counts.items())),
        "verification_path": str(verification_path),
        "verification_sha256": verification_sha,
        "accepted_path": str(accepted_path),
        "accepted_sha256": accepted_sha,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    if not accepted:
        raise RuntimeError("No correct samples survived filtering; refusing to train")
    atomic_write_json(success_path, summary)
    print(
        f"[filter] correct={correct_count}/{total}, retained={len(accepted)}, "
        f"coverage={len(correct_by_problem)}/{len(per_problem_raw)} -> {accepted_path}"
    )


if __name__ == "__main__":
    main()
