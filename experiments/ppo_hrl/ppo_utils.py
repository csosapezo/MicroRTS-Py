import argparse
from distutils.util import strtobool
import os
import time

import numpy as np
import torch


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-name',
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment',
    )
    parser.add_argument(
        '--gym-id',
        type=str,
        default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2.5e-4,
        help='the learning rate of the optimizer',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='seed of the experiment',
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=50000000,
        help='total timesteps of the experiments',
    )
    parser.add_argument(
        '--torch-deterministic',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`',
    )
    parser.add_argument(
        '--cuda',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help='if toggled, cuda will not be enabled by default',
    )
    parser.add_argument(
        '--capture-video',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help='whether to capture videos of the agent performances',
    )

    # Algorithm specific arguments
    parser.add_argument(
        '--partial-obs',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help='if toggled, the game will have partial observability',
    )
    parser.add_argument(
        '--n-minibatch',
        type=int,
        default=4,
        help='the number of mini batch',
    )
    parser.add_argument(
        '--num-bot-envs',
        type=int,
        default=0,
        help='the number of bot game environment; 16 bot envs means 16 games',
    )
    parser.add_argument(
        '--num-selfplay-envs',
        type=int, default=24,
        help='the number of self play envs; 16 self play envs means 8 games',
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=256,
        help='the number of steps per game environment',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='the discount factor gamma',
    )
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='the lambda for the general advantage estimation',
    )
    parser.add_argument(
        '--ent-coef',
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        '--vf-coef',
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='the maximum norm for the gradient clipping',
    )
    parser.add_argument(
        '--clip-coef',
        type=float,
        default=0.1,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        '--update-epochs',
        type=int,
        default=4,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        '--kle-stop',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help='If toggled,policy updates will be early stopped w.r.t target-kl',
    )
    parser.add_argument(
        '--kle-rollback',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help='If toggled,policy updates will roll back '
        'to previous policy if KL exceeds target-kl',
    )
    parser.add_argument(
        '--target-kl',
        type=float,
        default=0.03,
        help='the target-kl variable that is referred by --kl',
    )
    parser.add_argument(
        '--gae',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help='Use GAE for advantage computation',
    )
    parser.add_argument(
        '--norm-adv',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        '--anneal-lr',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        '--clip-vloss',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help='Toggles whether or not to use a clipped loss'
        'for the value function, as per the paper.',
    )
    parser.add_argument(
        '--num-models',
        type=int,
        default=100,
        help='the number of models saved',
    )
    parser.add_argument(
        '--last-step',
        type=int,
        default=0,
        help='last step (0 if no previous update)',
    )
    parser.add_argument(
        '--last-exp-name',
        type=str,
        help='name of the last experiment',
    )
    parser.add_argument(
        '--train-maps',
        nargs='+',
        default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during training',
    )
    parser.add_argument(
        '--eval-maps',
        nargs='+',
        default=["maps/16x16/basesWorkers16x16A.xml"],
        help='the list of maps used during evaluation',
    )

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    args.num_updates = args.total_timesteps // args.batch_size
    args.save_frequency = max(1, int(args.num_updates // args.num_models))
    # fmt: on
    return args


class EpochData:
    def __init__(
        self,
        n_steps,
        n_envs,
        h,
        w,
        obs_dims,
        action_space_dims,
        device,
        reward_weight=None,
        has_iam=False,
        iam_shape=None,
    ):
        self.obs = torch.zeros((n_steps, n_envs, h, w, obs_dims)).to(device)

        if action_space_dims > 1:
            self.actions = torch.zeros(
                (n_steps, n_envs, h * w, action_space_dims)
            ).to(device)
        else:
            self.actions = torch.zeros((n_steps, n_envs, h * w)).to(device)

        self.logprobs = torch.zeros((n_steps, n_envs)).to(device)
        self.rewards = torch.zeros((n_steps, n_envs)).to(device)
        self.dones = torch.zeros((n_steps, n_envs)).to(device)
        self.values = torch.zeros((n_steps, n_envs)).to(device)
        self.reward_weight = reward_weight

        if has_iam:
            self.invalid_action_masks = torch.zeros(
                (n_steps, n_envs) + iam_shape
            ).to(device)

        self.advantages = None
        self.returns = None

    def flatten(
        self,
        h,
        w,
        obs_dims,
        action_space_dims,
        has_iam=False,
        iam_shape=None,
    ):
        self.b_obs = self.obs.reshape(-1, h, w, obs_dims)
        self.b_logprobs = self.logprobs.reshape(-1)
        if action_space_dims > 1:
            self.b_actions = self.actions.reshape(-1, h * w, action_space_dims)
        else:
            self.b_actions = self.actions.reshape(-1, h * w)
        self.b_advantages = self.advantages.reshape(-1)
        self.b_returns = self.returns.reshape(-1)
        self.b_values = self.values.reshape(-1)

        if has_iam:
            self.b_invalid_action_masks = self.invalid_action_masks.reshape(
                (-1,) + iam_shape
            )


def get_rewards(infos, reward_weight):
    raw_rewards = np.array([info["raw_rewards"] for info in infos])
    return raw_rewards @ reward_weight
