"""Verify complete evaluation shards and compute per-problem pass@k estimates."""

from __future__ import annotations

import argparse
import json
import math
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
    write_immutable_jsonl,
)
from verify_math import verify_completion


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator without constructing binomial coefficients."""
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got n={n}, c={c}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n], got n={n}, k={k}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    log_miss = 0.0
    for index in range(k):
        log_miss += math.log(n - c - index) - math.log(n - index)
    return -math.expm1(log_miss)


def _discover(directory: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    success_files = sorted(directory.glob("shard-*.success.json"))
    if not success_files:
        raise FileNotFoundError(f"No completed evaluation shards in {directory}")
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in success_files]
    counts = {item["shard_count"] for item in metadata}
    if len(counts) != 1:
        raise ValueError(f"Mixed shard counts in {directory}: {counts}")
    count = counts.pop()
    indices = {item["shard_index"] for item in metadata}
    if indices != set(range(count)):
        raise ValueError(f"Incomplete eval shards: have {sorted(indices)}, need 0..{count - 1}")
    paths = [path.with_name(path.name.replace(".success.json", ".jsonl")) for path in success_files]
    return paths, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/restem.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--round", type=int, required=True, dest="model_round")
    args = parser.parse_args()

    config = load_experiment(args.config, args.experiment)
    if not 0 <= args.model_round <= config["rounds"]:
        raise ValueError(f"Model round must be in [0, {config['rounds']}]")
    raw_dir = generation_dir(config, "eval", args.model_round)
    raw_paths, raw_metadata = _discover(raw_dir)
    output_dir = experiment_dir(config) / "eval" / f"m{args.model_round}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_contract = {
        **provenance(config),
        "model_round": args.model_round,
        "verifier": config["dataset"]["verifier"],
        "verifier_timeout_seconds": config["filter"]["verifier_timeout_seconds"],
        "raw_contracts": sorted(item["generation_contract_sha256"] for item in raw_metadata),
        "raw_files": {str(path): file_sha256(path) for path in raw_paths},
    }
    contract_hash = canonical_sha256(score_contract)
    success_path = output_dir / "_SUCCESS.json"
    if success_path.exists():
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success.get("score_contract_sha256") != contract_hash:
            raise FileExistsError(f"Scoring contract changed for existing output: {output_dir}")
        print(f"[passk] M{args.model_round} already scored: {output_dir}")
        return

    samples: dict[str, dict[str, Any]] = {}
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_path in raw_paths:
        for row in iter_jsonl(raw_path):
            if row["sample_id"] in samples:
                raise ValueError(f"Duplicate sample_id across eval shards: {row['sample_id']}")
            samples[row["sample_id"]] = row
            by_problem[row["problem_id"]].append(row)
    expected_n = int(config["generation"]["eval"]["samples_per_problem"])
    bad_counts = {
        problem_id: len(rows)
        for problem_id, rows in by_problem.items()
        if len(rows) != expected_n
    }
    if bad_counts:
        raise ValueError(
            f"pass@k requires exactly n={expected_n}; {len(bad_counts)} problem(s) differ: "
            f"{list(sorted(bad_counts.items()))[:5]}"
        )

    sample_scores: list[dict[str, Any]] = []
    per_problem: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for problem_id in sorted(by_problem):
        rows = sorted(by_problem[problem_id], key=lambda row: row["sample_index"])
        correct = 0
        parsed = 0
        truncated = 0
        for row in rows:
            verification = verify_completion(
                config["dataset"]["verifier"],
                row["completion"],
                row["answer"],
                int(config["filter"]["verifier_timeout_seconds"]),
            )
            correct += int(verification["correct"])
            parsed += int(verification["parsed"])
            truncated += int(row["truncated"])
            totals[verification["failure_reason"] or "correct"] += 1
            sample_scores.append(
                {
                    "sample_id": row["sample_id"],
                    "problem_id": problem_id,
                    "sample_index": row["sample_index"],
                    "truncated": row["truncated"],
                    **verification,
                }
            )
        estimates = [pass_at_k(expected_n, correct, k) for k in range(1, expected_n + 1)]
        per_problem.append(
            {
                "problem_id": problem_id,
                "model_round": args.model_round,
                "n": expected_n,
                "c": correct,
                "parsed": parsed,
                "truncated": truncated,
                "subject": rows[0].get("subject"),
                "level": rows[0].get("level"),
                "pass_at_k": estimates,
            }
        )

    curve = [
        sum(row["pass_at_k"][index] for row in per_problem) / len(per_problem)
        for index in range(expected_n)
    ]
    sample_scores_path = output_dir / "sample_scores.jsonl"
    per_problem_path = output_dir / "per_problem.jsonl"
    sample_scores_sha = write_immutable_jsonl(sample_scores_path, sample_scores)
    per_problem_sha = write_immutable_jsonl(per_problem_path, per_problem)
    total_samples = len(sample_scores)
    summary = {
        **score_contract,
        "score_contract_sha256": contract_hash,
        "problem_count": len(per_problem),
        "samples_per_problem": expected_n,
        "total_samples": total_samples,
        "parse_rate": sum(row["parsed"] for row in per_problem) / total_samples,
        "correct_rate": sum(row["c"] for row in per_problem) / total_samples,
        "truncation_rate": sum(row["truncated"] for row in per_problem) / total_samples,
        "problems_solved_at_least_once": sum(row["c"] > 0 for row in per_problem),
        "failure_counts": dict(sorted(totals.items())),
        "k": list(range(1, expected_n + 1)),
        "pass_at_k": curve,
        "sample_scores_path": str(sample_scores_path),
        "sample_scores_sha256": sample_scores_sha,
        "per_problem_path": str(per_problem_path),
        "per_problem_sha256": per_problem_sha,
    }
    atomic_write_json(output_dir / "curve.json", summary)
    atomic_write_json(success_path, summary)
    print(
        f"[passk] M{args.model_round}: pass@1={curve[0]:.4f}, "
        f"pass@{expected_n}={curve[-1]:.4f}, solved={summary['problems_solved_at_least_once']}/"
        f"{len(per_problem)}"
    )


if __name__ == "__main__":
    main()
