from pathlib import Path
import argparse, importlib, json, zipfile
import torch, torch.nn as nn

_SB3 = {
    "ppo": "stable_baselines3.PPO",
    "maskable_ppo": "sb3_contrib.ppo_mask.MaskablePPO",
    "a2c": "stable_baselines3.A2C",
}

def _auto_algo(z: Path) -> str | None:
    with zipfile.ZipFile(z) as f:
        if "data/parameters.json" in f.namelist():
            return json.loads(f.read("data/parameters.json"))["algo"].lower()
    return None

def _load(path: Path, algo: str):
    m, c = _SB3[algo].rsplit(".", 1)
    return getattr(importlib.import_module(m), c).load(path, device="cpu")

class ScriptablePolicy(nn.Module):
    """Minimal subset of an SB3 policy that covers the forward pass."""
    def __init__(self, sb3_pol):
        super().__init__()
        self.feat = sb3_pol.features_extractor      # HexCNN (conv1+conv2)
        self.pi   = sb3_pol.mlp_extractor.policy_net[0]  # Linear(128→64)
        self.head = sb3_pol.action_net              # Linear(64→49)

    def forward(self, obs):                         # obs: (B,3,7,7)
        x = torch.tanh(self.feat(obs))
        x = torch.tanh(self.pi(x))
        return self.head(x)                         # logits (B,49)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--algo", choices=_SB3)
    a = ap.parse_args()

    algo = a.algo or _auto_algo(a.checkpoint)
    if algo not in _SB3:
        raise SystemExit("❌  Could not auto-detect algo – pass --algo")

    sb3_model = _load(a.checkpoint, algo)
    wrapper   = ScriptablePolicy(sb3_model.policy).eval()
    scripted  = torch.jit.script(wrapper)           # now succeeds :contentReference[oaicite:2]{index=2}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    scripted.save(a.out_dir / "policy.pt")
    print("✅  wrote", a.out_dir / "policy.pt")

if __name__ == "__main__":
    main()
