import gymnasium as gym
import numpy as np
from reil_hex_game.hex_engine.hex_engine import hexPosition

class HexEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int = 7):
        super().__init__()
        self.size = size
        self.game = hexPosition(size=size)

        # one flat action per board cell
        self.action_space = gym.spaces.Discrete(size * size)

        # 3-channel tensor = P1 stones, P-1 stones, “who-to-move”
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, size, size), dtype=np.float32
        )

    # -------------------------------------------------------------
    # Gym reset ---------------------------------------------------
    # -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """
        Required by Gymnasium: must return (observation, info).
        """
        super().reset(seed=seed)
        self.game.reset()
        return self._obs(), {}         # empty info-dict

    # -------------------------------------------------------------
    # Gym step ----------------------------------------------------
    # -------------------------------------------------------------
    def step(self, action: int):
        if action not in self._legal_scalar_moves():
            # Illegal ⇒ immediate loss (−1) and terminate
            reward, terminated, truncated = -1.0, True, False
            return self._obs(), reward, terminated, truncated, {}
 
        coord = self.game.scalar_to_coordinates(action)
        self.game.move(coord)                 # one legal move

        terminated = self.game.winner != 0    # game finished?
        truncated  = False                    # no time-limit
        reward = float(self.game.winner) if terminated else 0.0

        return self._obs(), reward, terminated, truncated, {}

    def action_masks(self):
        """Boolean mask – True where the move is **allowed**."""
        mask = np.zeros(self.size * self.size, dtype=bool)
        mask[self._legal_scalar_moves()] = True
        return mask

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def _obs(self):
        """
        3-channel tensor (C, H, W):
        0: player-1 stones
        1: player-(−1) stones
        2: “who-to-move” flag (all 1 if P1 to move, 0 otherwise)
        Counting stones - not `self.game.current_player` - avoids
        relying on a field that doesn’t exist in hexPosition.
        """
        board = np.asarray(self.game.board, dtype=np.float32)
        p1_layer  = (board ==  1).astype(np.float32)
        p_1_layer = (board == -1).astype(np.float32)

        flat = board.ravel()
        to_move = 1 if np.count_nonzero(flat == 1) == np.count_nonzero(flat == -1) else -1
        turn_layer = np.full_like(board, 1.0 if to_move == 1 else 0.0)

        return np.stack([p1_layer, p_1_layer, turn_layer], axis=0)
    
    def _flat_board(self):
        """Return the board as a flat 1-D NumPy array (shape = size²)."""
        return np.asarray(self.game.board, dtype=np.int8).ravel()

    def _legal_scalar_moves(self):
        """Indices of empty cells (0-valued) in 0 … size²-1 order."""
        return np.nonzero(self._flat_board() == 0)[0].tolist()

class OpponentWrapper(gym.Wrapper):
    """
    On the opponent's turn, delegate the move to `opponent(agent_view)`.
    `opponent` is a function (board, legal_moves) -> coordinate.
    """
    def __init__(self, env, opponent_fn):
        super().__init__(env)
        self.opponent_fn = opponent_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated:
            # opponent responds immediately
            board_mat = self.env.game.board
            legal = self.env.game.legal_moves()
            opp_coord = self.opponent_fn(board_mat, legal)
            self.env.game.move(opp_coord)
            # flip reward sign because turns alternate
            reward = -reward
            terminated = self.env.game.winner != 0
        return self.env._obs(), reward, terminated, truncated, info