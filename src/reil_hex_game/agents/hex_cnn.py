import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class HexCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 64):
        n_input_channels = observation_space.shape[0]
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # compute output size
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None])
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))
