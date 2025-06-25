# reil_hex_game/hex_engine/side_swap_wrapper.py
import gymnasium as gym
import numpy as np
import random

class SideSwapWrapper(gym.Wrapper):
    """
    At reset() randomly chooses which side the *learning* agent will play.
    * If self.agent_side ==  1 → wrapped env works as usual (agent starts).
    * If self.agent_side == -1 → opponent moves first; observations and
      rewards are flipped so the policy always 'thinks' it is P1.
    """
    def __init__(self, env, opponent_fn, prob_start_first: float = 0.5):
        super().__init__(env)
        self.opponent_fn = opponent_fn
        self.prob_start_first = prob_start_first
        self.agent_side = 1               # will be set at every reset()

    # ─────────────────────────────────────────────────────────────
    def reset(self, **kw):
        obs, info = self.env.reset(**kw)

        # 1. decide sides
        self.agent_side = 1 if random.random() < self.prob_start_first else -1

        # 2. if opponent starts, let it move immediately
        if self.agent_side == -1:
            board   = self.env.game.board
            legal   = self.env.game.legal_moves()
            opp_mv  = self.opponent_fn(board, legal)
            self.env.game.move(opp_mv)         # opponent makes first move
            obs = self._transform_obs(self.env._obs())   # flip perspective

        return obs, info

    # ─────────────────────────────────────────────────────────────
    def step(self, action):
        if self.agent_side == -1:
            # incoming action is in 'flipped' coordinates
            action = self._flip_scalar(action)

        obs, r, term, trunc, info = self.env.step(action)

        # opponent’s turn if not finished
        if not (term or trunc):
            board  = self.env.game.board
            legal  = self.env.game.legal_moves()
            opp_mv = self.opponent_fn(board, legal)
            self.env.game.move(opp_mv)
            # reward sign flips because opponent moved
            r = -r
            term = self.env.game.winner != 0
            obs = self.env._obs()

        # finally, if we are P-1, transform back to agent’s canonical view
        if self.agent_side == -1:
            r   = -r                         # win for P-1 => +1 for agent
            obs = self._transform_obs(obs)

        return obs, r, term, trunc, info

    # ─────────────────────────────────────────────────────────────
    # helpers
    def _flip_scalar(self, scalar_idx: int) -> int:
        """Mirror a 0..N²-1 index around the main diagonal."""
        size = self.env.size
        row, col = divmod(scalar_idx, size)
        flipped_idx = col * size + row
        return flipped_idx

    def _transform_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Swap channel-0 and channel-1 so agent always sees itself as 'red'
        (P1).  Also flip the 'who-to-move' flag.
        """
        obs = obs.copy()
        obs[[0, 1]] = obs[[1, 0]]          # swap stone layers
        obs[2]      = 1.0 - obs[2]         # invert turn map (1↔0)
        return obs

    # forward legality mask so ActionMasker still works
    def action_masks(self):
        mask = self.env.action_masks()
        if self.agent_side == -1:
            # mirror mask the same way we mirror observations
            flat_mask = mask.reshape(self.env.size, self.env.size).T.ravel()
            return flat_mask
        return mask
