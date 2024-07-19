from torch.distributions.categorical import Categorical
import torch.nn as nn


from agent_utils import Transpose, layer_init


class MacroAgent(nn.Module):
    def __init__(self, envs, mapsize=16 * 16):
        super(MacroAgent, self).__init__()
        self.mapsize = mapsize
        # (h, w, c) <- envs.observation_space.shape
        _, _, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            # n_envs, h, w, c -> n_envs, c, h, w
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 4 macro actions: move, harvest, produce, attack
        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(
                64, 32, 3,
                stride=2, padding=1, output_padding=1,
            )),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(
                32, 4, 3,
                stride=2, padding=1, output_padding=1,
            )),
            Transpose((0, 2, 3, 1)),
        )

        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )

    def get_action_and_value(
        self,
        x,
        action=None,
        device=None,
    ):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1, 4)

        categorical = Categorical(logits=grid_logits)
        action = categorical.sample()

        logprob = categorical.log_prob(action)
        entropy = categorical.entropy()

        logprob = logprob.view(-1, self.mapsize)
        entropy = entropy.view(-1, self.mapsize)
        action = action.view(-1, self.mapsize)

        return (
            action,
            logprob.sum(1),
            entropy.sum(1),
            self.critic(hidden),
        )

    def get_value(self, x):
        return self.critic(self.encoder(x))
