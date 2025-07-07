import gymnasium as gym
import numpy as np
import random

class SideSwapWrapper(gym.Wrapper):
    """
    A final, unified wrapper for the Hex environment that correctly handles
    two-player logic, perspective swapping, and detailed information logging
    for TensorBoard.
    """
    def __init__(self, env, opponent_fn, prob_start_first: float = 0.5):
        super().__init__(env)
        self.opponent_fn = opponent_fn
        self.prob_start_first = prob_start_first
        self.agent_side = 1

    @staticmethod
    def get_last_strategy():
        """
        Fetches the last used strategy name from the rule_based_helper module.
        This is crucial for detailed logging of the opponent's behavior.
        """
        try:
            from reil_hex_game.agents import rule_based_helper
            return rule_based_helper.LAST_STRATEGY_USED
        except (ImportError, AttributeError):
            return "unknown" # Fallback if not a rule-based agent

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.agent_side = 1 if random.random() < self.prob_start_first else -1
        self.unwrapped.agent_player_id = self.agent_side

        if self.agent_side == -1:
            board = self.env.game.board
            legal_moves = self.env.game.legal_moves()
            if legal_moves:
                opp_mv = self.opponent_fn(board, legal_moves)
                self.env.game.move(opp_mv)
            obs = self._transform_obs(self.env._obs())
        return obs, info

    def step(self, action):
        # --- 1. Agent's Move ---
        actual_action = self._flip_scalar(action) if self.agent_side == -1 else action
        obs, reward, terminated, truncated, info = self.env.step(actual_action)

        if terminated or truncated:
            final_reward = -reward if self.agent_side == -1 else reward
            if self.agent_side == -1:
                obs = self._transform_obs(obs)
            return obs, final_reward, terminated, truncated, info

        # --- 2. Opponent's Move ---
        board = self.env.game.board
        legal_moves = self.env.game.legal_moves()
        if not legal_moves:
            return obs, reward, True, truncated, info

        opp_mv = self.opponent_fn(board, legal_moves)
        self.env.game.move(opp_mv)

        # --- 3. Final State, Reward, and Info Calculation ---
        terminated = self.env.game.winner != 0
        final_obs = self.env._obs()

        # The base reward is the shaping value from the agent's move.
        final_reward = reward
        if terminated:
            # If the game ends, override with the definitive win/loss reward.
            final_reward = 1.0 if self.env.game.winner == self.agent_side else -1.0

        # --- 4. Logging and Perspective Transformation ---
        # The `info` dict from env.step already contains shaping/potential info.
        # We add the opponent's strategy info here.
        info['opponent_strategy'] = self.get_last_strategy()

        if self.agent_side == -1:
            final_obs = self._transform_obs(final_obs)
            final_reward = -final_reward

        return final_obs, final_reward, terminated, truncated, info

    def action_masks(self):
        mask = self.env.action_masks()
        if self.agent_side == -1:
            return mask.reshape(self.env.size, self.env.size).T.ravel()
        return mask

    def _flip_scalar(self, scalar_idx: int) -> int:
        size = self.env.size
        row, col = divmod(scalar_idx, size)
        return col * size + row

    def _transform_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        obs[[0, 1]] = obs[[1, 0]]
        obs[2] = 1.0 - obs[2]
        return obs
