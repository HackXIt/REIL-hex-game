import gymnasium as gym
from reil_hex_game.hex_engine.hex_engine import hexPosition

class HexEnv(gym.Env):
    def __init__(self, size=7):
        self.game = hexPosition(size=size)
        self.action_space = gym.spaces.Discrete(size*size)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(size, size), dtype=int)
    def reset(self, *, seed=None, options=None):
        self.game.reset()
        return self._obs(), {}
    def step(self, action_scalar):
        coord = self.game.scalar_to_coordinates(action_scalar)
        self.game.move(coord)
        done = self.game.winner != 0
        reward = float(self.game.winner)   # +1 / –1 / 0
        return self._obs(), reward, done, False, {}
    def _obs(self):
        return np.array(self.game.board, dtype=int)
