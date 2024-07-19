import torch
import torch.nn as nn


from agent_utils import CategoricalMasked, Transpose, layer_init


class MicroAgent(nn.Module):
    def __init__(self, envs, mapsize=16 * 16):
        super(MicroAgent, self).__init__()
        self.mapsize = mapsize
        # (h, w, c) <- envs.observation_space.shape
        _, _, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c + 1, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(
                64, 32, 3,
                stride=2, padding=1, output_padding=1,
            )),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(
                32, 78, 3,
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
        self.register_buffer("mask_value", torch.tensor(-1e8))

    def get_action_and_value(
        self,
        x,
        action=None,
        invalid_action_masks=None,
        envs=None,
        device=None,
    ):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1, envs.action_plane_space.nvec.sum())
        split_logits = torch.split(
            grid_logits,
            envs.action_plane_space.nvec.tolist(),
            dim=1,
        )

        if action is None:
            invalid_action_masks = invalid_action_masks.view(
                -1,
                invalid_action_masks.shape[-1],
            )
            split_invalid_action_masks = torch.split(
                invalid_action_masks,
                envs.action_plane_space.nvec.tolist(),
                dim=1,
            )
            multi_categoricals = [
                CategoricalMasked(
                    logits=logits,
                    masks=iam,
                    mask_value=self.mask_value,
                )
                for (logits, iam) in zip(
                    split_logits,
                    split_invalid_action_masks,
                )
            ]
            action = torch.stack(
                [categorical.sample() for categorical in multi_categoricals]
            )
        else:
            invalid_action_masks = invalid_action_masks.view(
                -1,
                invalid_action_masks.shape[-1],
            )

            action = action.view(-1, action.shape[-1]).T

            split_invalid_action_masks = torch.split(
                invalid_action_masks,
                envs.action_plane_space.nvec.tolist(),
                dim=1,
            )

            multi_categoricals = [
                CategoricalMasked(
                    logits=logits,
                    masks=iam,
                    mask_value=self.mask_value,
                )
                for (logits, iam)
                in zip(split_logits, split_invalid_action_masks)
            ]

        logprob = torch.stack([
            categorical.log_prob(a)
            for a, categorical
            in zip(action, multi_categoricals)
        ])

        entropy = torch.stack([
            categorical.entropy()
            for categorical
            in multi_categoricals
        ])

        num_predicted_parameters = len(envs.action_plane_space.nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)

        return (
            action,
            logprob.sum(1).sum(1),
            entropy.sum(1).sum(1),
            invalid_action_masks,
            self.critic(hidden),
        )

    def get_value(self, x):
        return self.critic(self.encoder(x))
