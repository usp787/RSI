"""Sharded, resumable vLLM sampling for ReST-EM training or pass@k evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    append_jsonl,
    atomic_write_json,
    canonical_sha256,
    checkpoint_dir,
    file_sha256,
    generation_shard_path,
    iter_jsonl,
    load_experiment,
    manifest_dir,
    provenance,
    select_deterministic_subset,
    stable_int,
    text_sha256,
)
from prompts import render_prompt


def _model_source(config: dict[str, Any], model_round: int) -> tuple[str, str | None]:
    if model_round == 0:
        return config["model"]["base_id"], config["model"]["revision"]
    merged = checkpoint_dir(config, model_round) / "merged"
    success = checkpoint_dir(config, model_round) / "_SUCCESS.json"
    if not merged.is_dir() or not success.exists():
        raise FileNotFoundError(f"Missing completed M{model_round} checkpoint: {merged}")
    return str(merged), None


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for row in iter_jsonl(path):
        sample_id = row["sample_id"]
        if sample_id in existing:
            raise ValueError(f"Duplicate sample_id in resumable output {path}: {sample_id}")
        existing[sample_id] = row
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/restem.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--round", type=int, required=True, dest="model_round")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()

    config = load_experiment(args.config, args.experiment)
    if args.model_round < 0 or args.model_round > config["rounds"]:
        raise ValueError(f"Round must be in [0, {config['rounds']}]")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    if args.phase == "train" and args.model_round >= config["rounds"]:
        raise ValueError("The final model round has no training-generation phase")

    manifest_path = manifest_dir(config) / ("train.jsonl" if args.phase == "train" else "eval.jsonl")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run prepare_data.py first: {manifest_path}")
    all_problems = list(iter_jsonl(manifest_path))
    limit_key = "max_train_problems" if args.phase == "train" else "max_eval_problems"
    selected = select_deterministic_subset(
        all_problems, config.get(limit_key), config["seed"], f"{args.experiment}:{args.phase}"
    )
    shard_problems = [
        row for position, row in enumerate(selected) if position % args.shard_count == args.shard_index
    ]
    if not shard_problems:
        raise ValueError(
            f"Shard {args.shard_index}/{args.shard_count} is empty; use fewer shards than problems"
        )

    sampling = config["generation"][args.phase]
    samples_per_problem = int(sampling["samples_per_problem"])
    model_path, model_revision = _model_source(config, args.model_round)
    few_shot = bool(config["dataset"]["prompt"][f"{args.phase}_few_shot"])
    output_path = generation_shard_path(
        config, args.phase, args.model_round, args.shard_index, args.shard_count
    )
    metadata_path = output_path.with_suffix(".meta.json")
    success_path = output_path.with_suffix(".success.json")
    metadata = {
        **provenance(config),
        "phase": args.phase,
        "model_round": args.model_round,
        "model_path": model_path,
        "model_revision": model_revision,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "sampling": sampling,
        "few_shot": few_shot,
        "selected_problem_ids_sha256": canonical_sha256([row["problem_id"] for row in selected]),
        "selected_problem_count": len(selected),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_problem_count": len(shard_problems),
        "expected_sample_count": len(shard_problems) * samples_per_problem,
    }
    contract_hash = canonical_sha256(metadata)
    metadata["generation_contract_sha256"] = contract_hash

    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("generation_contract_sha256") != contract_hash:
            raise FileExistsError(f"Generation contract changed for existing shard: {output_path}")
    else:
        atomic_write_json(metadata_path, metadata)

    existing = _load_existing(output_path)
    if success_path.exists():
        if len(existing) != metadata["expected_sample_count"]:
            raise ValueError(f"Success marker exists but sample count is wrong: {output_path}")
        print(f"[generate] already complete: {output_path}")
        return

    from vllm import LLM, SamplingParams

    model_config = config["model"]
    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": model_config["dtype"],
        "tensor_parallel_size": model_config["tensor_parallel_size"],
        "gpu_memory_utilization": model_config["gpu_memory_utilization"],
        "max_model_len": model_config["max_model_len"],
        "seed": config["seed"],
        "trust_remote_code": False,
    }
    if model_revision is not None:
        llm_kwargs["revision"] = model_revision
        llm_kwargs["tokenizer_revision"] = model_revision
    print(f"[generate] loading M{args.model_round}: {model_path}")
    llm = LLM(**llm_kwargs)

    batch_size = int(sampling.get("request_batch_size", config["generation"]["request_batch_size"]))
    for offset in range(0, len(shard_problems), batch_size):
        batch = shard_problems[offset : offset + batch_size]
        prompts = [render_prompt(row["problem"], few_shot=few_shot) for row in batch]
        params = []
        for row in batch:
            seed_parts = [config["seed"], args.experiment, args.phase]
            if args.phase == "train":
                seed_parts.append(args.model_round)
            request_seed = stable_int(*seed_parts, row["problem_id"])
            params.append(
                SamplingParams(
                    n=samples_per_problem,
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    top_k=int(sampling["top_k"]),
                    max_tokens=int(sampling["max_tokens"]),
                    seed=request_seed,
                    skip_special_tokens=True,
                )
            )
        outputs = llm.generate(prompts=prompts, sampling_params=params, use_tqdm=True)
        if len(outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} requests for batch of {len(batch)}")

        new_rows: list[dict[str, Any]] = []
        for problem, prompt, request_output, request_params in zip(batch, prompts, outputs, params):
            candidates = sorted(request_output.outputs, key=lambda item: item.index)
            if len(candidates) != samples_per_problem:
                raise RuntimeError(
                    f"Expected {samples_per_problem} outputs for {problem['problem_id']}, "
                    f"got {len(candidates)}"
                )
            for candidate in candidates:
                sample_id = text_sha256(
                    f"{contract_hash}:{problem['problem_id']}:{candidate.index}"
                )
                if sample_id in existing:
                    continue
                row = {
                    "sample_id": sample_id,
                    "problem_id": problem["problem_id"],
                    "dataset": config["dataset_key"],
                    "phase": args.phase,
                    "model_round": args.model_round,
                    "sample_index": candidate.index,
                    "request_seed": request_params.seed,
                    "prompt": prompt,
                    "prompt_sha256": text_sha256(prompt),
                    "completion": candidate.text,
                    "completion_sha256": text_sha256(candidate.text),
                    "answer": problem["answer"],
                    "subject": problem.get("subject"),
                    "level": problem.get("level"),
                    "finish_reason": candidate.finish_reason,
                    "output_tokens": len(candidate.token_ids),
                    "truncated": candidate.finish_reason == "length",
                    "generation_contract_sha256": contract_hash,
                }
                new_rows.append(row)
                existing[sample_id] = row
        append_jsonl(output_path, new_rows)
        print(
            f"[generate] shard {args.shard_index}: {len(existing)}/"
            f"{metadata['expected_sample_count']} samples"
        )

    if len(existing) != metadata["expected_sample_count"]:
        raise RuntimeError(
            f"Incomplete shard: expected {metadata['expected_sample_count']}, found {len(existing)}"
        )
    completion = {
        **metadata,
        "actual_sample_count": len(existing),
        "output_sha256": file_sha256(output_path),
    }
    atomic_write_json(success_path, completion)
    print(f"[generate] complete: {output_path}")


if __name__ == "__main__":
    main()
