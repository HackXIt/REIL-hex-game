# REIL-Hex-Game

This project focuses on creating a reinforcement learning agent to play the game of Hex.

It was conducted as the term assignment in the Reinforcement Learning course at the [University of Applied Science Technikum Vienna](https://www.technikum-wien.at/) as part of the masters programme in Artificial Intelligence, under the supervision of [Dr. Rosana De Oliveira Gomes](https://www.technikum-wien.at/en/staff/rosana-de-oliveira-gomes/).

-----

## Introduction

This project implements a reinforcement learning environment for the game of Hex. It includes various agents, training scripts, and a visualization tool to play against the trained models. The primary goal is to train an effective agent using algorithms like PPO and A2C.

-----

## The Game of Hex

Hex is a two-player abstract strategy board game in which players attempt to connect opposite sides of a hexagonal grid. The game is played on a rhombus-shaped board, and players take turns placing their colored stones on any empty cell. The first player to form a connected path of their stones from one side of the board to the opposite side wins. The game can never end in a draw, and because of the first-player advantage, the pie rule (swap rule) is often implemented. In this project, the [`SideSwapWrapper`](src/reil_hex_game/hex_engine/side_swap_wrapper.py) class randomly chooses which side the learning agent will play to mitigate this advantage.

-----

## ⚙️ Project Structure

The project is divided into two main directories: [`src/`](src/) and [`scripts/`](scripts/).

### `src/`

The `src` directory contains the core logic of the Hex game and the agents.

  * `reil_hex_game/`: This is the main package.
      * `agents/`: This directory contains different types of agents that can play the game:
          * `rule_based_agent.py`: A simple agent that follows a set of predefined logic based strategies. 
          * `rule_based_helper.py`: Helper functions for the rule-based agent.
          * `rule_based_v1_agent.py`, `rule_based_v2_agent.py`, `rule_based_v3_agent.py`, `rule_based_v4_agent.py`: Different implementations of rule-based agents with v3 and v4 being most advanced versions, using weights to prioritize certain moves.
          * `sb3_agent.py`: An agent that uses a pre-trained Stable Baselines3 model.
          * `tscript_agent.py`: An agent that uses a loaded TorchScript model.
      * `hex_engine/`: This directory contains the game engine and environment.
          * `hex_engine.py`: The core Hex game logic, handling the board state, moves, and win conditions.
          * `hex_env.py`: The Gymnasium environment for the Hex game, which is used for training reinforcement learning agents.
          * `hex_pygame/`: The visualization of the game using Pygame.

### `scripts/`

The `scripts` directory contains scripts for training, exporting, and managing models.

-----

## 🚀 Usage

### Command-Line Interface (CLI)

The project provides a CLI to play the game. You can start a game using the `reil-hex-game` command.

```bash
uv run reil-hex-game --mode hvm --agent rule_based_v4 --use-pygame
```

The CLI supports different modes (`hvh`, `hvm`, `mvm`) and allows you to specify the agent to play against. You can also enable the Pygame GUI with the `--use-pygame` flag. For a full list of options, run `reil-hex-game --help`.

<details>
<summary>Available CLI Options</summary>

```shell
usage: reil-hex-game [-h] [--mode {hvh,hvm,mvm}] [--auto] [--rate RATE] [--board-size BOARD_SIZE] [--use-pygame] [--agent AGENT [AGENT ...]]
                     [--human-player {1,2}] [-i]

Play the Hex board game from the command line.

options:
  -h, --help            show this help message and exit
  --mode {hvh,hvm,mvm}  Game mode: hvh (human vs human), hvm (human vs machine), mvm (machine vs machine)
  --auto                Start machine-vs-machine in continuous auto-play.
  --rate RATE           Auto-play speed in moves / second (default 3).
  --board-size BOARD_SIZE
                        Board side length (2-26, default 7)
  --use-pygame          Enable pygame GUI
  --agent AGENT [AGENT ...]
                        Built-in agent name or module:attr path. Accepts one or two values - when two are given they are used as agent 1 and agent 2. In hvm
                        mode the second value is ignored.
  --human-player {1,2}  For hvm mode: 1=white, 2=black (default 1)
  -i, --interactive     Prompt for options interactively (default when no flags are given)
```
</details>

### Training Script

The main training script is [`scripts/train_alg_hex.py`](scripts/train_alg_hex.py). This script acts as a dispatcher, calling the appropriate training script based on the `--algo` argument.

```bash
python -m scripts.train_alg_hex --algo ppo --board-size 7 --timesteps 1e6
```

<details>
<summary>Available CLI Options</summary>

```shell
usage: train_alg_hex [-h] --algo {ppo,a2c} [--board-size BOARD_SIZE] [--timesteps TIMESTEPS] [--num-envs NUM_ENVS] [--run-name RUN_NAME]
                     [--save-dir SAVE_DIR] [--device DEVICE] [--resume] [--video-every VIDEO_EVERY] [--video-len VIDEO_LEN] [--video-eval]
                     [--rule-fraction RULE_FRACTION]

Dispatch to an algorithm-specific Hex training script

options:
  -h, --help            show this help message and exit
  --algo {ppo,a2c}      Training algorithm back-end
  --board-size BOARD_SIZE
                        Hex board side length (default 7)
  --timesteps TIMESTEPS
                        Total environment steps for on-policy algos or frames for off-policy (default 1 000 000)
  --num-envs NUM_ENVS   Parallel Gymnasium environments (vectorised)
  --run-name RUN_NAME   Custom identifier inside the output directory
  --save-dir SAVE_DIR   Root directory where checkpoints & logs go
  --device DEVICE       'cpu', 'cuda', or 'auto' (SB3 style)
  --resume              Continue training from the latest checkpoint in --save-dir
  --video-every VIDEO_EVERY
                        Record a rollout every N environment steps (0 = disable video)
  --video-len VIDEO_LEN
                        Number of frames per recorded video
  --video-eval          Render evaluation games (rgb_array) so they appear in the video stream too.
  --rule-fraction RULE_FRACTION
                        Fraction of parallel envs that include the rule-based opponent.
```
</details>

-----

## 📦 Dependency Management with `uv`

This project uses [`uv`](https://docs.astral.sh/uv/) for fast dependency and project management. The project's dependencies are defined in the [`pyproject.toml`](pyproject.toml) file.

To get started, create a virtual environment and install the required packages using `uv`:

```bash
# Create a virtual environment
uv venv

# Update the projects environment
uv sync

# Install dependencies
uv add <package-name>
# Remove dependencies
uv remove <package-name>
# Install packages into the virtual environment directly (package will not be added to the project)
uv pip install <package-name>
```

The `pyproject.toml` file also defines a script for running the games main module directly. You can use `uv run` to execute it:

```bash
uv run reil-hex-game <... CLI options>
```

-----

## 🤖 Training

The project includes training scripts for both **A2C (Advantage Actor-Critic)** and **PPO (Proximal Policy Optimization)** algorithms. These scripts are responsible for setting up the reinforcement learning environment, configuring the agent's policy network, and running the training loop.

### A2C Training

The [`a2c_train.py`](scripts/a2c_train.py) script trains an agent using the A2C algorithm. Here's a breakdown of the training process:

-   **Opponent Mix**: The training environment is configured to use a mix of different opponents to improve the agent's robustness. This includes various versions of the rule-based agent (`rule_based_v3_agent`, `rule_based_v4_agent`) and a random agent. The proportion of each opponent type can be adjusted.
-   **Vectorized Environments**: The script uses `SubprocVecEnv` to run multiple environments in parallel, which significantly speeds up the training process by collecting more experience in less time.
-   **Action Masking**: To ensure the agent only selects legal moves, the `ActionMasker` wrapper is used. This wrapper provides a mask of legal actions to the agent at each step.
-   **Callbacks**: Several callbacks are used during training:
    -   `CheckpointCallback`: Saves the model at regular intervals.
    -   `FastEval`: A modified evaluation callback that periodically evaluates the agent's performance against a strong opponent (`rule_based_v4_agent`).
    -   `StrategyTBCallback`: Logs the usage of different opponent strategies to TensorBoard for analysis.

### PPO Training

The [`ppo_train.py`](scripts/ppo_train.py) script trains an agent using the PPO algorithm, which is known for its stability and performance. The process is similar to A2C training but with a key difference in the algorithm used:

-   **Maskable PPO**: This script uses `MaskablePPO` from the `sb3-contrib` library. This version of PPO is specifically designed to handle environments with invalid actions, making it a perfect fit for Hex where not all board positions are playable at any given time.
-   **Opponent and Environment Setup**: The opponent mix, vectorized environments, and action masking are set up in the same way as the A2C training script.
-   **Side Swap Wrapper**: To mitigate the first-player advantage in Hex, the `SideSwapWrapper` is used. This wrapper randomly decides which side the learning agent will play at the beginning of each episode.
-   **Policy and Callbacks**: The PPO agent also uses the `CnnPolicy` with the `HexCNN` feature extractor. The same set of callbacks (`CheckpointCallback`, `FastEval`, `StrategyTBCallback`) are used to monitor and save the training progress.
-   **Hyperparameters**: The PPO script uses a different set of hyperparameters (e.g., `n_steps`, `batch_size`, `n_epochs`) that are more suitable for the PPO algorithm.

#### HexCNN Feature Extractor

A custom feature extractor, [`HexCNN`](src/reil_hex_game/agents/hex_cnn.py), is used to process the game's state. The board is represented as a 3-channel tensor (player 1's stones, player -1's stones, and a turn indicator). The `HexCNN` consists of two convolutional layers followed by a linear layer. This architecture allows the agent to learn spatial patterns and relationships between the stones on the board, which is crucial for developing effective strategies in Hex.

-----

## Model Export

### TorchScript Export

The [`scripts/export_tscript.py`](scripts/export_tscript.py) script exports a trained Stable Baselines3 model to a TorchScript file (`policy.pt`). This allows the model to be used in a Python environment without the Stable Baselines3 dependency.

```bash
python -m scripts.export_tscript --checkpoint <path_to_checkpoint>.zip --out-dir <output_directory>
```
<details>
<summary>Available CLI Options</summary>

```shell
usage: export_tscript.py [-h] --checkpoint CHECKPOINT --out-dir OUT_DIR [--algo {ppo,maskable_ppo,a2c}]

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
  --out-dir OUT_DIR
  --algo {ppo,maskable_ppo,a2c}
```
</details>

### Importing the TorchScript Model

The exported TorchScript model can be loaded and used by the [`tscript_agent.py`](src/reil_hex_game/agents/tscript_agent.py). The agent loads the `policy.pt` file and uses it to select moves. The file needs to be placed into the same directory as the agent script.

-----

## Rule-Based Agent

The project includes a rule-based agent that uses a set of predefined strategies to play the game. The logic for these strategies is implemented in the [`rule_based_helper.py`](src/reil_hex_game/agents/rule_based_helper.py) file. The agent can be used as a baseline for comparison against reinforcement learning agents.

A documentation about these strategies can be found in the [`docs/hex-game-strategies.md`](docs/hex-game-strategies.md) file. The rule-based agent is designed to play against human players or other agents, providing a simple but better than random opponent.

The various strategies implemented in the rule-based agent include:
- `take_center`: Suggests playing in the center of the board.
- `extend_own_chain`: Focuses on extending existing chains of the player's stones.
- `shortest_connection`: Finds the shortest path to victory.
- `break_opponent_bridge`: Identifies and breaks the opponent's bridges.
- `protect_own_chain_from_cut`: Protects the player's chains from being cut by the opponent.
- `create_double_threat`: Creates a situation where the player has two winning moves available.
- `make_own_bridge`: Suggests moves that create a bridge for the player.
- `mild_block_threat`: Blocks the opponent's threats without being too aggressive.
- `advance_toward_goal`: Moves towards the player's goal.
- `block_aligned_opponent_path`: Blocks the opponent's most direct path.

-----

## Hex Game Visualization

The game can be visualized using Pygame. The visualization logic is located in [`src/reil_hex_game/hex_engine/hex_pygame/`](src/reil_hex_game/hex_engine/hex_pygame/). This module is based on the [`hex-py`](https://github.com/parappayo/hex-py/) project by [Parappayo](https://github.com/parappayo/). Special credit goes to the original author for the implementation. The code was modified to fit with the pre-existing `hex_engine` logic and synchronize with the game state.

The visualization displays the game board, the players' moves, and the winning path when the game is over. It provides an interactive way to play against the trained agents.

In the visualization, when playing human vs. machine or human vs. human, the game board is displayed, and players can click on the hexagonal cells to place their stones. The game will automatically switch turns between players.

In machine vs. machine, the game will run continuously, displaying the moves made by both agents. The speed of the game can be adjusted using the `--rate` option in the CLI. One can also pause or continue playing by pressing the space bar. Pressing ENTER will step through the moves.

-----

## Reference to Term Paper

For a more in-depth look at the project, including the theoretical background, implementation details, and experimental results, please refer to the term paper located in the [`hex-game-paper/`](hex-game-paper/) submodule.

If there is broken submodule link, please visit the repository directly: [https://github.com/hackxit/hex-game-paper](https://github.com/hackxit/hex-game-paper)