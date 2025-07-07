import subprocess
import json
import re
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

def get_steps_from_filename(p: Path) -> int:
        """Extracts the step number from a checkpoint filename."""
        # Use regex to find the number in 'model_100000_steps.zip'
        match = re.search(r"model_(\d+)_steps", p.stem)
        if match:
            return int(match.group(1))
        return 0 # Fallback for unexpected filenames

def find_latest_model(run_name: str) -> Path:
    """
    Finds the latest model checkpoint in a given run directory.
    It correctly handles 'model_..._steps.zip' and 'final.zip'.
    """
    run_dir = Path("runs") / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found for self-play opponent: {run_dir}")

    final_model = run_dir / "final.zip"
    checkpoints = list(run_dir.glob("model_*.zip"))

    # If final.zip exists, it is considered the latest model.
    if final_model.exists():
        print(f"   Found 'final.zip' model for {run_name}.")
        return final_model

    if not checkpoints:
        raise FileNotFoundError(f"No models ('final.zip' or 'model_*.zip') found in {run_dir}")

    # Sort checkpoints by the extracted step number in descending order
    checkpoints.sort(key=get_steps_from_filename, reverse=True)
    
    latest_model = checkpoints[0]
    print(f"   Found latest checkpoint: {latest_model.name}")
    return latest_model

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
