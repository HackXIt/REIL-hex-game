#!/usr/bin/env python3
"""
scripts/train_alg_hex.py
========================
Dispatcher CLI: route to the concrete training script that matches --algo.

Example
-------
$ python -m scripts.train_alg_hex --algo ppo --board-size 7 --timesteps 3e6
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
from datetime import datetime

# Map short name → module (without .py)
_ALGOS = {
    "ppo":        "ppo_train",
    "a2c":        "a2c_train",
    # "reinforce":  "reinforce_train",
    # "alphazero":  "az_train",
}

def parse_timesteps(x: str) -> int:
    """Accept int literals or scientific notation like 3e6."""
    return int(float(x.replace("_", "")))   # allows 1_000_000 too

# -------------------------------------------------------------------------
# Argument handling
# -------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_alg_hex",
        description="Dispatch to an algorithm-specific Hex training script"
    )
    p.add_argument("--algo", required=True, choices=_ALGOS,
                   help="Training algorithm back-end")
    p.add_argument("--board-size", type=int, default=7,
                   help="Hex board side length (default 7)")
    p.add_argument("--timesteps", type=parse_timesteps, default=1_000_000,
                   help="Total environment steps for on-policy algos or "
                        "frames for off-policy (default 1 000 000)")
    p.add_argument("--num-envs", type=int, default=16,
                   help="Parallel Gymnasium environments (vectorised)")
    p.add_argument("--run-name", default=None,
                   help="Custom identifier inside the output directory")
    p.add_argument("--save-dir", default="runs",
                   help="Root directory where checkpoints & logs go")
    p.add_argument("--device", default="auto",
                   help="'cpu', 'cuda', or 'auto' (SB3 style)")
    p.add_argument("--resume", action="store_true",
                   help="Continue training from the latest checkpoint "
                        "in --save-dir")
    p.add_argument("--video-every", type=int, default=0,
                   help="Record a rollout every N environment steps (0 = disable video)")
    p.add_argument("--video-len", type=int, default=200,
                   help="Number of frames per recorded video")
    p.add_argument("--video-eval", action="store_true",
                   help="Render evaluation games (rgb_array) so they appear in the video stream too.")
    p.add_argument("--rule-fraction", type=float, default=0.5, help="Fraction of parallel envs that include the rule-based opponent.")
    return p.parse_args(argv)

# -------------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Resolve trainer module
    mod_name = _ALGOS[args.algo]
    try:
        trainer = importlib.import_module(f"scripts.{mod_name}")
    except ModuleNotFoundError:
        print(f"[ERROR] scripts/{mod_name}.py not found – create it first.",
              file=sys.stderr)
        sys.exit(1)

    if not hasattr(trainer, "train"):
        print(f"[ERROR] {mod_name}.py must expose a train(args) function.",
              file=sys.stderr)
        sys.exit(1)

    # Add a sensible default run-name
    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.algo}_{args.board_size}x{args.board_size}_{ts}"

    trainer.train(args)      # hand over control

if __name__ == "__main__":      # allow "python -m scripts.train_alg_hex"
    main()
