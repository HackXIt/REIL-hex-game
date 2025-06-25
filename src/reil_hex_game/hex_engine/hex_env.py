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

    _STRATEGY_WEIGHTS = {
        "take_center"                : 1,
        "extend_own_chain"           : 3,
        "break_opponent_bridge"      : 3,
        "protect_own_chain_from_cut" : 3,
        "create_double_threat"       : 4,
        "shortest_connection"        : 4,
        "make_own_bridge"            : 3,
        "mild_block_threat"          : 2,
        "advance_toward_goal"        : 2,
        "block_aligned_opponent_path": 3,
    }
    _SHAPING_SCALE = 0.01
    _GAMMA = 0.99

    def __init__(self, size: int = 7, render_mode: str | None = None, prob_start_first: float = 0.5):
        super().__init__()
        assert render_mode in {None, *self.metadata["render_modes"]}
        self.render_mode = render_mode
        self.size = size
        self.prob_start_first = prob_start_first
        self.agent_side = 1
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
            self._surface = pygame.Surface((1000, 1000)).convert()
            if self.render_mode == "human":
                # visible window
                self._screen = pygame.display.set_mode(
                    self._surface.get_size(),
                    pygame.SCALED | pygame.DOUBLEBUF
                )
        elif self.render_mode == "rgb_array":
            pygame.init()
            # make an off-screen surface that does NOT depend on display format
            self._surface = pygame.Surface((1000, 1000), flags=pygame.SRCALPHA)

    # -------------------------------------------------------------
    # Gym reset ---------------------------------------------------
    # -------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """
        Required by Gymnasium: must return (observation, info).
        """
        super().reset(seed=seed, options=options)
        self.game.reset()
        self._last_potential = self._potential()   # initialise ϕ(s₀)
        return self._obs(), {}

    # -------------------------------------------------------------
    # Gym step ----------------------------------------------------
    # -------------------------------------------------------------
    def step(self, action: int):
        if action not in self._legal_scalar_moves():
            # Illegal ⇒ immediate loss (−1) and terminate
            reward, terminated, truncated = -1.0, True, False
            return self._obs(), reward, terminated, truncated, {}
 
        coord = self.game.scalar_to_coordinates(action)
        self.game.move(coord)

        terminated = self.game.winner != 0
        truncated  = False
        sparse_r   = float(self.game.winner) if terminated else 0.0

        # ----- dense shaping -------------------------------------------------
        new_pot = self._potential()
        shaping = self._GAMMA * new_pot - self._last_potential
        self._last_potential = new_pot

        reward = sparse_r + shaping      # ← final reward returned to PPO
        info = {"shaping": shaping, "potential": new_pot}
        return self._obs(), reward, terminated, truncated, info

    def action_masks(self):
        """Boolean mask – True where the move is **allowed**."""
        mask = np.zeros(self.size * self.size, dtype=bool)
        mask[self._legal_scalar_moves()] = True
        return mask

    # -------------------------------------------------------------
    # Gym render --------------------------------------------------
    # -------------------------------------------------------------
    def render(self, *_, **__):
        mode = self.render_mode or "rgb_array"
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode={mode!r}")

        # ─── build / cache a GameState that matches the board size ────────
        if not hasattr(self, "_gs"):
            from reil_hex_game.hex_engine.hex_pygame import game_state
            self._gs = game_state.GameState()
            self._gs.board_width_tiles  = self.size
            self._gs.board_height_tiles = self.size
            self._gs.hex_tile_size      = 32
            self._gs.generate_board()                       # re-create grid
        gs = self._gs

        # ─── update tile colours to reflect self.game.board ───────────────
        empty_c, p1_c, p2_c = (
            gs.empty_hex_colour,
            gs.player_colour[0],
            gs.player_colour[1],
        )

        for (row, col), tile in gs.hex_grid.tiles.items():
            if row < self.size and col < self.size:
                cell = self.game.board[row][col]
                tile.colour = p1_c if cell == 1 else p2_c if cell == -1 else empty_c
            else:
                tile.colour = empty_c                       # unused tiles

        # ─── compute and cache winning path once ─────────────────────
        if self.game.winner != 0 and gs.solution is None:
            gs.solution = gs.find_solution()

        # ─── draw onto the cached off-screen surface ──────────────────────
        from reil_hex_game.hex_engine.hex_pygame import game_draw
        game_draw.draw_frame(self._surface, gs, flip=False)

        if mode == "human":                                 # on-screen debug
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
            pygame.event.pump()
            return None

        # rgb_array – turn Surface → np.ndarray(H, W, 3) uint8
        frame = np.transpose(
            pygame.surfarray.array3d(self._surface), (1, 0, 2)
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
        1: player-(-1) stones
        2: “who-to-move” flag (all 1 if P1 to move, 0 otherwise)
        Counting stones - not `self.game.current_player` - avoids
        relying on a field that doesn't exist in hexPosition.
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
    
    def _potential(self) -> float:
        """
        Dense heuristic φ(s) built from *all* rule-based strategies.
        Positive is good for the current player.
        """
        from reil_hex_game.agents import rule_based_helper as rh
        player = 1 if self._to_move() == 1 else -1
        board  = self.game.board
        action_set = self.game.legal_moves()   # whatever the helper functions need

        # --- 1. path-length term (same as before) --------------------------
        path = rh.shortest_connection_path(board, player)
        path_len = self.size * 2 if path is None else len(path) - 1
        pot = -self._SHAPING_SCALE * float(path_len)

        # --- 2. add weighted bonuses for every strategy --------------------
        # Map helper names → callables (import once for speed)
        if not hasattr(self, "_strat_fns"):
            from reil_hex_game.agents import rule_based_helper as rh
            self._strat_fns = {
                "take_center"                : rh.take_center,
                "extend_own_chain"           : rh.extend_own_chain,
                "break_opponent_bridge"      : rh.break_opponent_bridge,
                "protect_own_chain_from_cut" : rh.protect_own_chain_from_cut,
                "create_double_threat"       : rh.create_double_threat,
                "shortest_connection"        : rh.shortest_connection,
                "make_own_bridge"            : rh.make_own_bridge,
                "mild_block_threat"          : rh.mild_block_threat,
                "advance_toward_goal"        : rh.advance_toward_goal,
                "block_aligned_opponent_path": rh.block_aligned_opponent_path,
            }

        for name, fn in self._strat_fns.items():
            if fn(board, action_set, player):          # returns bool
                pot += 0.05 * self._STRATEGY_WEIGHTS[name]  # 1-to-4 → 0.05-0.20

        # --- clamp for stability ------------------------------------------
        return max(-1.0, min(1.0, pot))

    def _to_move(self):
        flat = np.ravel(self.game.board)
        return 1 if np.count_nonzero(flat == 1) == np.count_nonzero(flat == -1) else -1

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
    
    def action_masks(self): # Forward in-case of ordering issue in environment wrappers
        """Return the action mask for the current player."""
        return self.env.action_masks()