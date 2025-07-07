from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

# This global variable will be set by the training script
_model = None
_model_path = None

def set_opponent_model_path(path: str):
    """Dynamically sets the path to the opponent model file."""
    global _model_path, _model
    _model_path = Path(path)
    _model = None # Reset model so it reloads with the new path

def _load_model():
    """Lazy-loads the PPO model once."""
    global _model
    if _model is None:
        if _model_path is None or not _model_path.exists():
            raise FileNotFoundError(
                "Opponent model path not set or file not found. "
                "Use set_opponent_model_path() before training."
            )
        _model = PPO.load(_model_path, device="cpu")
    return _model

def ppo_opponent_agent(board, action_set):
    """
    An agent function that uses a pre-trained PPO model to select an action.
    """
    model = _load_model()

    # Reconstruct the 3-channel observation from the board state
    obs_board = np.asarray(board, dtype=np.float32)
    p1 = (obs_board == 1).astype(np.float32)
    p_1 = (obs_board == -1).astype(np.float32)
    is_p1_turn = np.count_nonzero(p1) == np.count_nonzero(p_1)
    turn = np.full_like(obs_board, 1.0 if is_p1_turn else 0.0)
    obs = np.stack([p1, p_1, turn])[None]  # Add batch dimension

    # Create the action mask
    mask = np.zeros((1, len(board) ** 2), dtype=bool)
    for r, c in action_set:
        mask[0, r * len(board) + c] = True

    # Get action from the model
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    
    # Convert scalar action back to coordinate
    size = len(board)
    return (int(action[0] // size), int(action[0] % size))
