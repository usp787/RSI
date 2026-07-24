"""Create matched pass@k curves, paired bootstrap intervals, and coverage sets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from common import (
    atomic_write_json,
    canonical_sha256,
    experiment_dir,
    iter_jsonl,
    load_experiment,
    provenance,
)


def declared_rounds(configured_final_round: int, submitted_final_round: int) -> list[int]:
    """Return every declared model round and reject truncated reports."""
    if submitted_final_round != configured_final_round:
        raise ValueError(
            "Report final round must exactly match the experiment configuration: "
            f"submitted={submitted_final_round}, configured={configured_final_round}"
        )
    return list(range(configured_final_round + 1))


def validate_output_name(output_name: str) -> str:
    """Keep report recovery outputs inside the experiment artifact directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_name):
        raise ValueError(
            "--output-name must be one safe directory name using letters, numbers, '.', '_', or '-'"
        )
    return output_name


def _bootstrap_curves(matrix, replicates: int, seed: int, batch_size: int = 200):
    import numpy as np

    rng = np.random.default_rng(seed)
    problem_count = matrix.shape[0]
    probabilities = np.full(problem_count, 1.0 / problem_count)
    chunks = []
    remaining = replicates
    while remaining:
        size = min(batch_size, remaining)
        counts = rng.multinomial(problem_count, probabilities, size=size)
        chunks.append((counts @ matrix) / problem_count)
        remaining -= size
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/restem.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--final-round", type=int, required=True)
    parser.add_argument(
        "--output-name",
        default="report",
        help="safe output directory name under the experiment artifact directory",
    )
    args = parser.parse_args()

    config = load_experiment(args.config, args.experiment)
    rounds = declared_rounds(int(config["rounds"]), args.final_round)
    output_name = validate_output_name(args.output_name)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows_by_round: dict[int, dict[str, dict[str, Any]]] = {}
    score_contracts: dict[int, str] = {}
    score_code_commits: dict[int, str] = {}
    for model_round in rounds:
        directory = experiment_dir(config) / "eval" / f"m{model_round}"
        success_path = directory / "_SUCCESS.json"
        per_problem_path = directory / "per_problem.jsonl"
        if not success_path.exists() or not per_problem_path.exists():
            raise FileNotFoundError(f"M{model_round} evaluation is incomplete: {directory}")
        success = json.loads(success_path.read_text(encoding="utf-8"))
        score_contracts[model_round] = success["score_contract_sha256"]
        score_code_commits[model_round] = success["code_commit"]
        round_rows = {row["problem_id"]: row for row in iter_jsonl(per_problem_path)}
        rows_by_round[model_round] = round_rows

    source_code_commits = set(score_code_commits.values())
    if len(source_code_commits) != 1:
        raise ValueError(f"Evaluation scores were produced by mixed code commits: {score_code_commits}")

    base_ids = set(rows_by_round[0])
    for model_round, round_rows in rows_by_round.items():
        if set(round_rows) != base_ids:
            raise ValueError(f"M{model_round} problem IDs do not exactly match M0")
    problem_ids = sorted(base_ids)
    n_values = {
        row["n"] for round_rows in rows_by_round.values() for row in round_rows.values()
    }
    if len(n_values) != 1:
        raise ValueError(f"Evaluation sample budgets differ across models: {n_values}")
    max_k = n_values.pop()
    k_values = np.arange(1, max_k + 1)
    matrices = {
        model_round: np.asarray(
            [rows_by_round[model_round][problem_id]["pass_at_k"] for problem_id in problem_ids],
            dtype=float,
        )
        for model_round in rounds
    }
    report_config = config["report"]
    replicates = int(report_config["bootstrap_replicates"])
    seed = int(report_config["bootstrap_seed"])
    curves: dict[int, dict[str, Any]] = {}
    base_matrix = matrices[0]
    base_bootstrap = _bootstrap_curves(base_matrix, replicates, seed)
    for model_round in rounds:
        matrix = matrices[model_round]
        curve = matrix.mean(axis=0)
        if model_round == 0:
            bootstrap = base_bootstrap
            delta = np.zeros(max_k)
            delta_bootstrap = np.zeros((replicates, max_k))
        else:
            difference = matrix - base_matrix
            delta_bootstrap = _bootstrap_curves(
                difference, replicates, seed + 1000 + model_round
            )
            bootstrap = _bootstrap_curves(matrix, replicates, seed + model_round)
            delta = difference.mean(axis=0)
        curves[model_round] = {
            "pass_at_k": curve,
            "ci_lower": np.quantile(bootstrap, 0.025, axis=0),
            "ci_upper": np.quantile(bootstrap, 0.975, axis=0),
            "delta_vs_m0": delta,
            "delta_ci_lower": np.quantile(delta_bootstrap, 0.025, axis=0),
            "delta_ci_upper": np.quantile(delta_bootstrap, 0.975, axis=0),
        }

    emphasized = [int(k) for k in report_config["emphasized_k"] if int(k) <= max_k]
    table_rows: list[dict[str, Any]] = []
    for model_round in rounds:
        for k in emphasized:
            index = k - 1
            item = curves[model_round]
            table_rows.append(
                {
                    "model": f"M{model_round}",
                    "k": k,
                    "pass_at_k": float(item["pass_at_k"][index]),
                    "ci_lower": float(item["ci_lower"][index]),
                    "ci_upper": float(item["ci_upper"][index]),
                    "delta_vs_m0": float(item["delta_vs_m0"][index]),
                    "delta_ci_lower": float(item["delta_ci_lower"][index]),
                    "delta_ci_upper": float(item["delta_ci_upper"][index]),
                }
            )

    coverage: dict[str, Any] = {}
    base_solved = {problem_id for problem_id in problem_ids if rows_by_round[0][problem_id]["c"] > 0}
    for model_round in rounds[1:]:
        current_solved = {
            problem_id for problem_id in problem_ids if rows_by_round[model_round][problem_id]["c"] > 0
        }
        coverage[f"M{model_round}_vs_M0"] = {
            "both": sorted(base_solved & current_solved),
            "m0_only": sorted(base_solved - current_solved),
            "iterated_only": sorted(current_solved - base_solved),
            "neither": sorted(set(problem_ids) - (base_solved | current_solved)),
        }

    breakdowns: dict[str, Any] = {}
    for dimension in ("subject", "level"):
        groups: dict[str, list[int]] = {}
        for index, problem_id in enumerate(problem_ids):
            value = rows_by_round[0][problem_id].get(dimension)
            if value is not None and str(value).lower() != "unknown":
                groups.setdefault(str(value), []).append(index)
        if not groups:
            continue
        breakdowns[dimension] = {}
        for group, indices in sorted(groups.items()):
            breakdowns[dimension][group] = {
                "problem_count": len(indices),
                "models": {
                    f"M{model_round}": {
                        f"pass@{k}": float(matrices[model_round][indices, k - 1].mean())
                        for k in emphasized
                    }
                    for model_round in rounds
                },
            }

    output_dir = experiment_dir(config) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_contract = {
        **provenance(config),
        "rounds": rounds,
        "score_contracts": score_contracts,
        "source_score_code_commit": source_code_commits.pop(),
        "report": report_config,
    }
    contract_hash = canonical_sha256(report_contract)
    success_path = output_dir / "_SUCCESS.json"
    if success_path.exists():
        existing = json.loads(success_path.read_text(encoding="utf-8"))
        if existing.get("report_contract_sha256") != contract_hash:
            raise FileExistsError(f"Report contract changed for existing output: {output_dir}")
        print(f"[report] already complete: {output_dir}")
        return
    summary = {
        **report_contract,
        "report_contract_sha256": contract_hash,
        "mode": config["mode"],
        "problem_count": len(problem_ids),
        "samples_per_problem": max_k,
        "warning": (
            "Smoke-test output is infrastructure validation only; do not interpret scientifically."
            if config["mode"] == "smoke"
            else None
        ),
        "emphasized_results": table_rows,
        "curves": {
            f"M{model_round}": {
                key: value.tolist() for key, value in curve.items()
            }
            for model_round, curve in curves.items()
        },
        "coverage_sets": coverage,
        "breakdowns": breakdowns,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    with (output_dir / "passk_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    figure, axis = plt.subplots(figsize=(8, 5))
    for model_round in rounds:
        item = curves[model_round]
        axis.plot(k_values, item["pass_at_k"], label=f"M{model_round}", linewidth=2)
    axis.set_xscale("log", base=2)
    axis.set_xticks(emphasized)
    axis.set_xticklabels([str(k) for k in emphasized])
    axis.set_xlabel("k")
    axis.set_ylabel("pass@k")
    axis.set_title(f"{args.experiment}: matched pass@k")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "passk.png", dpi=180)
    plt.close(figure)
    atomic_write_json(success_path, summary)
    print(f"[report] wrote matched curves and coverage sets -> {output_dir}")


if __name__ == "__main__":
    main()
