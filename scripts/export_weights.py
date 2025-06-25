# scripts/export_weights.py  (overwrite the previous file)
from __future__ import annotations
from pathlib import Path
import argparse, importlib, zipfile, json, sys
import numpy as np

_SB3_MAP = {
    "ppo": "stable_baselines3.PPO",
    "maskable_ppo": "sb3_contrib.ppo_mask.MaskablePPO",
    "a2c": "stable_baselines3.A2C",
}

# ---------- helper ---------------------------------------------------
def _detect_algo(zip_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            if "data/parameters.json" in zf.namelist():
                meta = json.loads(zf.read("data/parameters.json"))
                return meta.get("algo", "").lower()
    except Exception as e:
        print(f"[export] auto-detect failed: {e}", file=sys.stderr)
    return None

def _load_model(zip_path: Path, algo_hint: str | None):
    algo_name = algo_hint or _detect_algo(zip_path)
    if algo_name not in _SB3_MAP:
        raise ValueError(f"Unknown algo {algo_name!r}. Use --algo to set it.")
    module, clsname = _SB3_MAP[algo_name].rsplit(".", 1)
    cls = getattr(importlib.import_module(module), clsname)
    print(f"🔹 loading {clsname} checkpoint …")
    return cls.load(zip_path, device="cpu")

# ---------- export ---------------------------------------------------
def export(checkpoint_file: Path, out_dir: Path, algo_hint: str | None):
    model = _load_model(checkpoint_file, algo_hint)
    cnn   = model.policy.features_extractor.cnn           # conv(3→32)
    head  = model.policy.action_net                       # Linear(64→49)

    # ---- locate the hidden Linear(32→64) safely --------------------
    try:
        hidden = model.policy.mlp_extractor.policy_net[0]        # SB3 ≤ 2.1
    except AttributeError:
        # SB3 dev branch (latent_dim_pi attr exists but is an int)
        hidden = list(model.policy.mlp_extractor.policy_net)[0]  # generic

    weights = {
        "conv_w": cnn[0].weight.detach().numpy(),
        "conv_b": cnn[0].bias.detach().numpy(),
        "hid_w" : hidden.weight.detach().numpy(),
        "hid_b" : hidden.bias.detach().numpy(),
        "fc_w"  : head.weight.detach().numpy(),
        "fc_b"  : head.bias.detach().numpy(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "weights.npz", **weights)
    kb = (out_dir / "weights.npz").stat().st_size / 1024
    print(f"✅ wrote {out_dir/'weights.npz'} ({kb:.1f} KB)")

# ---------- CLI ------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--algo", choices=_SB3_MAP.keys())
    args = p.parse_args()
    export(args.checkpoint, args.out_dir, args.algo)
