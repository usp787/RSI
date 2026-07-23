"""ReST-EM Improve step: completion-only LoRA SFT, reset from M0 each round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    atomic_write_json,
    canonical_sha256,
    checkpoint_dir,
    experiment_dir,
    file_sha256,
    iter_jsonl,
    load_experiment,
    manifest_dir,
    provenance,
    select_deterministic_subset,
)
from prompts import render_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/restem.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--round", type=int, required=True, dest="generator_round")
    args = parser.parse_args()

    config = load_experiment(args.config, args.experiment)
    if not 0 <= args.generator_round < config["rounds"]:
        raise ValueError(f"Generator round must be in [0, {config['rounds'] - 1}]")
    model_round = args.generator_round + 1
    sft_dir = experiment_dir(config) / "sft" / f"round-{args.generator_round}"
    accepted_path = sft_dir / "accepted.jsonl"
    filter_success = sft_dir / "_SUCCESS.json"
    validation_path = manifest_dir(config) / "validation.jsonl"
    if not accepted_path.exists() or not filter_success.exists():
        raise FileNotFoundError(f"Completed filtered corpus is required: {sft_dir}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Fixed validation manifest is required: {validation_path}")

    output_dir = checkpoint_dir(config, model_round)
    trainer_dir = output_dir / "trainer"
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    success_path = output_dir / "_SUCCESS.json"
    training_contract = {
        **provenance(config),
        "algorithm": "restem-reset-to-m0",
        "generator_round": args.generator_round,
        "output_model_round": model_round,
        "initial_model": config["model"]["base_id"],
        "initial_model_revision": config["model"]["revision"],
        "accepted_sha256": file_sha256(accepted_path),
        "validation_sha256": file_sha256(validation_path),
        "sft": config["sft"],
    }
    contract_hash = canonical_sha256(training_contract)
    if success_path.exists():
        success = json.loads(success_path.read_text(encoding="utf-8"))
        if success.get("training_contract_sha256") != contract_hash:
            raise FileExistsError(f"Training contract changed for existing M{model_round}: {output_dir}")
        print(f"[sft] M{model_round} already complete: {merged_dir}")
        return

    accepted = list(iter_jsonl(accepted_path))
    if not accepted:
        raise ValueError(f"Accepted corpus is empty: {accepted_path}")
    train_rows = [
        {
            "prompt": row["prompt"],
            "completion": row["completion"],
            "sample_id": row["sample_id"],
        }
        for row in accepted
    ]
    validation = list(iter_jsonl(validation_path))
    validation = select_deterministic_subset(
        validation,
        config.get("max_eval_problems"),
        config["seed"],
        f"{args.experiment}:sft-validation",
    )
    few_shot = bool(config["dataset"]["prompt"]["train_few_shot"])

    def validation_completion(row: dict[str, Any]) -> str:
        if config["dataset_key"] != "gsm8k":
            return row["reference_solution"]
        rationale = row["reference_solution"].rsplit("####", maxsplit=1)[0].rstrip()
        return f"{rationale}\nTherefore, \\boxed{{{row['answer']}}}."

    validation_rows = [
        {
            "prompt": render_prompt(row["problem"], few_shot=few_shot),
            "completion": validation_completion(row),
            "problem_id": row["problem_id"],
        }
        for row in validation
    ]

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    base_id = config["model"]["base_id"]
    base_revision = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        base_id, revision=base_revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_sequence_tokens = 0
    target_completion_tokens = 0
    examples_exceeding_max_length = 0
    for row in train_rows:
        prompt_tokens = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        completion_tokens = len(
            tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        )
        sequence_tokens = prompt_tokens + completion_tokens + 1
        total_sequence_tokens += sequence_tokens
        target_completion_tokens += completion_tokens + 1
        examples_exceeding_max_length += int(sequence_tokens > int(config["sft"]["max_length"]))

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        revision=base_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False

    sft = config["sft"]
    lora_values = dict(sft["lora"])
    peft_config = LoraConfig(task_type="CAUSAL_LM", **lora_values)
    sft_args = SFTConfig(
        output_dir=str(trainer_dir),
        max_length=int(sft["max_length"]),
        completion_only_loss=True,
        packing=bool(sft["packing"]),
        learning_rate=float(sft["learning_rate"]),
        num_train_epochs=float(sft["num_train_epochs"]),
        max_steps=int(sft["max_steps"]),
        per_device_train_batch_size=int(sft["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(sft["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(sft["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(sft["gradient_checkpointing"]),
        logging_steps=int(sft["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(sft["eval_steps"]),
        save_strategy="steps",
        save_steps=int(sft["save_steps"]),
        save_total_limit=int(sft["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=float(sft["warmup_ratio"]),
        weight_decay=float(sft["weight_decay"]),
        bf16=bool(sft["bf16"]),
        tf32=bool(sft["tf32"]),
        optim="adamw_torch_fused",
        seed=int(config["seed"] + model_round),
        data_seed=int(config["seed"] + model_round),
        report_to=[],
        logging_first_step=True,
        save_safetensors=True,
        remove_unused_columns=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    last_checkpoint = get_last_checkpoint(str(trainer_dir)) if trainer_dir.exists() else None
    if last_checkpoint:
        print(f"[sft] resuming M{model_round} training from {last_checkpoint}")
    result = trainer.train(resume_from_checkpoint=last_checkpoint)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(merged_dir)

    metadata = {
        **training_contract,
        "training_contract_sha256": contract_hash,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "total_sequence_tokens_before_truncation": total_sequence_tokens,
        "target_completion_tokens_before_truncation": target_completion_tokens,
        "examples_exceeding_max_length": examples_exceeding_max_length,
        "effective_batch_size": (
            int(sft["per_device_train_batch_size"])
            * int(sft["gradient_accumulation_steps"])
        ),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "global_step": trainer.state.global_step,
        "training_metrics": result.metrics,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
    }
    atomic_write_json(output_dir / "training_metadata.json", metadata)
    atomic_write_json(success_path, metadata)
    print(f"[sft] complete M{model_round}: {merged_dir}")


if __name__ == "__main__":
    main()
