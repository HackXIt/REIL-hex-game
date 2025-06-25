import gymnasium as gym
import numpy as np
from reil_hex_game.hex_engine.hex_engine import hexPosition
import pygame
from pygame import surfarray

class HexEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, size: int = 7, render_mode: str | None = None):
        super().__init__()
        assert render_mode in {None, *self.metadata["render_modes"]}
        self.render_mode = render_mode
        self.size = size
        self.game = hexPosition(size=size)

        # one flat action per board cell
        self.action_space = gym.spaces.Discrete(size * size)

        # 3-channel tensor = P1 stones, P-1 stones, “who-to-move”
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, size, size), dtype=np.float32
        )

        # -------- pygame init (only once) -----------------------------
        if self.render_mode == "human":
            pygame.init()
            # off-screen buffer; reused every call
            self._surface = pygame.Surface((800, 800)).convert()
            if self.render_mode == "human":
                # visible window
                self._screen = pygame.display.set_mode(
                    self._surface.get_size(),
                    pygame.SCALED | pygame.DOUBLEBUF
                )
        elif self.render_mode == "rgb_array":
            pygame.init()
            # make an off-screen surface that does NOT depend on display format
            self._surface = pygame.Surface((800, 800), flags=pygame.SRCALPHA)

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
    # Gym render --------------------------------------------------
    # -------------------------------------------------------------
    def render(self):
        if self.render_mode is None:
            raise ValueError("render_mode was None, set it in __init__")

        # draw current board onto self._surface via your helper modules
        from reil_hex_game.hex_engine.hex_pygame import (
            game_state, game_draw
        )
        gs = game_state.GameState()
        gs.board = [row[:] for row in self.game.board]  # copy current board
        game_draw.draw_frame(self._surface, gs, flip=False)         # renders one frame

        if self.render_mode == "human":
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
            pygame.event.pump()   # keeps window responsive
            return None
        else:  # "rgb_array"
            # (W,H,3) → transpose to (H,W,3); copy to contiguous uint8
            frame = np.transpose(
                surfarray.array3d(self._surface), (1, 0, 2)
            ).copy()
            return frame

    # -------------------------------------------------------------
    # Gym close ---------------------------------------------------
    # -------------------------------------------------------------
    def close(self):
        if self.render_mode is not None:
            pygame.quit()

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
        return [self.game.coordinate_to_scalar(rc) for rc in self.game.legal_moves()]

class OpponentWrapper(gym.Wrapper):
    """
    On the opponent's turn, delegate the move to `opponent(agent_view)`.
    `opponent` is a function (board, legal_moves) -> coordinate.
    """
    def __init__(self, env, opponent_fn):
        super().__init__(env)
        self.opponent_fn = opponent_fn

    @staticmethod
    def get_last_strategy():
        try:
            from reil_hex_game.agents import LAST_STRATEGY_USED
        except ImportError:
            # If the agent does not define LAST_STRATEGY_USED, return None
            return None
        return LAST_STRATEGY_USED

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            # opponent responds immediately
            board_mat = self.env.game.board
            legal = self.env.game.legal_moves()
            opp_coord = self.opponent_fn(board_mat, legal)
            self.env.game.move(opp_coord)
            last_strategy = self.get_last_strategy()
            if last_strategy:
                info["opponent_strategy"] = last_strategy
            else:
                info["opponent_strategy"] = "unknown"
            info["opponent_move"] = opp_coord
            # flip reward sign because turns alternate
            reward = -reward
            terminated = self.env.game.winner != 0
        return self.env._obs(), reward, terminated, truncated, info