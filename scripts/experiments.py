import subprocess
import json
from pathlib import Path

# ==============================================================================
# DEFINE YOUR EXPERIMENTAL PLAN HERE
# ==============================================================================
# Each dictionary defines one training run. The script will execute them in order.
# - name: A unique identifier for the run. This creates a folder in `runs/`.
# - wrapper_type: 'opponent' (fixed perspective) or 'two_player' (side-swapped).
# - rule_fraction: % of opponents that are rule-based (v3/v4). The rest are random.
# - timesteps: Total training steps for this run.
# - self_play_from: (Optional) For self-play, specify the 'name' of the previous
#   run to use as the opponent. The script will find the latest model from that run.
# ==============================================================================
EXPERIMENTS = [
    {
        "name": "E1_PPO_FixedSide_vs_Random",
        "wrapper_type": "opponent",
        "rule_fraction": 0.0,
        "timesteps": 500_000,
    },
    {
        "name": "E2_PPO_SideSwap_vs_Random",
        "wrapper_type": "two_player",
        "rule_fraction": 0.0,
        "timesteps": 500_000,
    },
    {
        "name": "E3_PPO_Curriculum_50_Rules",
        "wrapper_type": "two_player",
        "rule_fraction": 0.5,
        "timesteps": 500_000,
    },
    {
        "name": "E4_PPO_Curriculum_90_Rules",
        "wrapper_type": "two_player",
        "rule_fraction": 0.9,
        "timesteps": 500_000,
    },
    {
        "name": "E5_PPO_SelfPlay_vs_E4",
        "wrapper_type": "two_player",
        "rule_fraction": 0.0,  # All opponents will be the previous best model
        "timesteps": 500_000,
        "self_play_from": "E4_PPO_Curriculum_90_Rules", # Use the best model from E4
    },
]

def find_latest_model(run_name: str) -> Path:
    """Finds the latest model checkpoint in a given run directory."""
    run_dir = Path("runs") / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found for self-play opponent: {run_dir}")

    checkpoints = list(run_dir.glob("model_*.zip"))
    if not checkpoints:
        final_model = run_dir / "final.zip"
        if final_model.exists():
            return final_model
        raise FileNotFoundError(f"No models found in {run_dir}")
    
    # Sort by the number in the filename to find the latest
    checkpoints.sort(key=lambda f: int(f.stem.split('_')[-1]))
    return checkpoints[-1]

def main():
    """Iterates through the experiment plan and runs each training session."""
    print(f"Starting experiment suite: {len(EXPERIMENTS)} runs planned.")
    print("=" * 70)

    for i, config in enumerate(EXPERIMENTS):
        print(f"\n▶️  Running Experiment {i+1}/{len(EXPERIMENTS)}: {config['name']}")
        print(f"   Config: {json.dumps(config, indent=2)}")
        print("-" * 70)

        # Base command using the dispatcher script
        command = [
            "python", "-m", "scripts.train_alg_hex",
            "--algo", "ppo",
            "--run-name", config["name"],
            "--wrapper-type", config["wrapper_type"],
            "--rule-fraction", str(config["rule_fraction"]),
            "--timesteps", str(config["timesteps"]),
        ]

        # Handle self-play logic
        opponent_model_path = None
        if "self_play_from" in config:
            try:
                opponent_model_path = find_latest_model(config["self_play_from"])
                print(f"   Found self-play opponent model: {opponent_model_path}")
                command.extend(["--opponent-model-path", str(opponent_model_path)])
            except FileNotFoundError as e:
                print(f"   ❌ ERROR: Could not run self-play experiment. {e}")
                continue

        try:
            # Execute the training command
            subprocess.run(" ".join(command), shell=True, check=True)
            print(f"\n✅ Experiment '{config['name']}' completed successfully.")
            print("=" * 70)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Experiment '{config['name']}' failed with error code {e.returncode}.")
            print("   Stopping the experiment suite.")
            break
        except KeyboardInterrupt:
            print("\n🛑 Experiment suite interrupted by user.")
            break

if __name__ == "__main__":
    main()
