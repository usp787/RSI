"""Fail-fast cluster checks that do not download or execute a model."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import subprocess
from pathlib import Path

from common import load_experiment


EXPECTED = {
    "torch": "2.8.0",
    "transformers": "4.57.6",
    "trl": "0.24.0",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "datasets": "4.5.0",
    "vllm": "0.11.0",
    "math-verify": "0.9.0",
    "PyYAML": "6.0.3",
    "matplotlib": "3.10.8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/restem.yaml")
    parser.add_argument("--experiment", default="smoke_gsm8k_3b")
    args = parser.parse_args()
    config = load_experiment(args.config, args.experiment)

    mismatches = {
        package: (expected, importlib.metadata.version(package))
        for package, expected in EXPECTED.items()
        if importlib.metadata.version(package) != expected
    }
    if mismatches:
        raise RuntimeError(f"Pinned package mismatch: {mismatches}")
    subprocess.run(["python", "-m", "pip", "check"], check=True)

    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")
    for module_name in ("torch", "transformers", "trl", "datasets", "vllm", "math_verify"):
        module = importlib.import_module(module_name)
        module_path = str(getattr(module, "__file__", ""))
        if "/.local/" in module_path or "\\.local\\" in module_path:
            raise RuntimeError(f"Host user-site leakage: {module_name} imported from {module_path}")
        print(f"[preflight] {module_name}: {module_path}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated Slurm GPU job")
    device_name = torch.cuda.get_device_name(0)
    if "H200" not in device_name.upper():
        raise RuntimeError(f"Expected an H200 allocation, found: {device_name}")
    print(f"[preflight] GPU: {device_name}; CUDA: {torch.version.cuda}")

    rsi_root = Path(os.environ["RSI_ROOT"]).resolve()
    for key, configured in config["storage"].items():
        path = Path(configured).resolve()
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        if rsi_root not in (path, *path.parents):
            raise RuntimeError(f"{key} escapes RSI_ROOT: {path}")
        print(f"[preflight] writable {key}: {path}")
    print("[preflight] environment, GPU, imports, config, and scratch paths are valid")


if __name__ == "__main__":
    main()
