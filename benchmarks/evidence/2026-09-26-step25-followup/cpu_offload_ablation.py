#!/usr/bin/env python3
"""Diagnostic monkeypatch only: run TinyMix with QKV spectral metrics on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import tiger_optim.tiger as tiger


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mac-step25"))
import diagnose_step25  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "cpu_offload"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "cpu_offload":
        original = tiger._spectral_dispersion_chunks

        def cpu_offload(chunks, low_band, high_band):
            if (
                not chunks or chunks[0].device.type != "mps"
                or len({chunk.numel() for chunk in chunks}) != 1
            ):
                return original(chunks, low_band, high_band)
            packed = torch.stack([chunk.reshape(-1).detach() for chunk in chunks]).to("cpu")
            measured = original(tuple(packed.unbind(0)), low_band, high_band)
            result = torch.stack([torch.stack(row) for row in measured]).to(chunks[0].device)
            return [tuple(result[i, j] for j in range(3)) for i in range(len(chunks))]

        tiger._spectral_dispersion_chunks = cpu_offload

    sys.argv = [str(HERE.parent / "2026-09-26-mac-step25/diagnose_step25.py"),
                "--variant", "full", "--steps", "29", "--output", str(args.output)]
    diagnose_step25.main()
    payload = json.loads(args.output.read_text())
    payload["spectral_mode"] = args.mode
    payload["ablation_script_sha256"] = sha256(Path(__file__))
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
