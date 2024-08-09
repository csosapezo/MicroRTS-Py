# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from gym_microrts import microrts_ai  # noqa
from macro_agent import MacroAgent
from micro_agent import MicroAgent
from ppo import MicroRTSStatsRecorder
from ppo_sota import Agent
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')

    # Algorithm specific arguments
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, the game will have partial observability')
    parser.add_argument('--num-steps', type=int, default=256,
        help='the number of steps per game environment')
    parser.add_argument("--agent-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument("--agent2-model-path", type=str, default="gym-microrts-static-files/agent_sota.pt",
        help="the path to the agent's model")
    parser.add_argument('--ai', type=str, default="",
        help='the opponent AI to evaluate against')
    parser.add_argument('--model-type', type=str, default=f"ppo_gridnet", choices=["ppo_gridnet_large", "ppo_gridnet"],
        help='the output path of the leaderboard csv')
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    if args.ai:
        args.num_bot_envs, args.num_selfplay_envs = 1, 0
    else:
        args.num_bot_envs, args.num_selfplay_envs = 0, 2
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    ais = []
    if args.ai:
        ais = [eval(f"microrts_ai.{args.ai}")]
    envs = MicroRTSGridModeVecEnv(
        num_bot_envs=len(ais),
        num_selfplay_envs=args.num_selfplay_envs,
        partial_obs=args.partial_obs,
        max_steps=5000,
        render_theme=2,
        ai2s=ais,
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs,
            f"videos/{experiment_name}",
            record_video_trigger=lambda x: x % 100000 == 0,
            video_length=2000,
        )
    assert isinstance(
        envs.action_space,
        MultiDiscrete,
    ), "only MultiDiscrete action space is supported"

    macro_agent = MacroAgent(envs).to(device)
    attack_agent = MicroAgent(envs).to(device)
    harvest_agent = MicroAgent(envs).to(device)
    move_agent = MicroAgent(envs).to(device)
    produce_agent = MicroAgent(envs).to(device)

    agent2 = Agent(envs).to(device)

    # ALGO Logic: Storage for epoch data
    mapsize = 16 * 16
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # CRASH AND RESUME LOGIC:
    starting_update = 1
    macro_agent.load_state_dict(torch.load(
        f"{args.agent_model_path}/macro_agent.pt",
        map_location=device,
    ))
    macro_agent.eval()

    attack_agent.load_state_dict(torch.load(
        f"{args.agent_model_path}/attack_agent.pt",
        map_location=device,
    ))
    attack_agent.eval()

    harvest_agent.load_state_dict(torch.load(
        f"{args.agent_model_path}/harvest_agent.pt",
        map_location=device,
    ))
    harvest_agent.eval()

    move_agent.load_state_dict(torch.load(
        f"{args.agent_model_path}/move_agent.pt",
        map_location=device,
    ))
    move_agent.eval()

    produce_agent.load_state_dict(torch.load(
        f"{args.agent_model_path}/produce_agent.pt",
        map_location=device,
    ))
    produce_agent.eval()

    agents = [move_agent, attack_agent, harvest_agent, produce_agent]

    if not args.ai:
        agent2.load_state_dict(torch.load(
            args.agent2_model_path,
            map_location=device,
        ))
        agent2.eval()

    wins = 0
    losses = 0
    ties = 0

    for update in range(starting_update, args.num_updates + 1):
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            envs.render()
            global_step += 1 * args.num_envs
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                invalid_action_masks = torch.tensor(
                    np.array(envs.get_action_mask())
                ).to(device)

                if args.ai:
                    m_action, _, _, _ = macro_agent.get_action_and_value(
                        next_obs,
                        device=device,
                    )

                    m_action = m_action.cpu() * envs.get_source_unit_mask()
                    m_action = m_action.view(args.num_envs, 16, 16).numpy()

                    micro_action = None

                    for num_agent, agent in enumerate(agents):

                        action_mask = torch.tensor(np.where(
                            m_action == num_agent,
                            1,
                            0,
                        ))
                        action_mask = torch.unsqueeze(action_mask, -1)
                        action_mask = action_mask.to(device)
                        micro_obs = torch.cat([next_obs, action_mask], -1)

                        # action -> 24, 256, 7
                        action, _, _, _, _ = agent.get_action_and_value(
                            micro_obs,
                            envs=envs,
                            invalid_action_masks=invalid_action_masks,
                            device=device,
                        )

                        action_mask = action_mask.view(args.num_envs, 256, 1)
                        action = action * action_mask.expand(action.size())

                        if micro_action is None:
                            micro_action = action
                        else:
                            micro_action += action

                    action = micro_action

                else:
                    p1_obs = next_obs[::2]
                    p2_obs = next_obs[1::2]
                    p1_mask = invalid_action_masks[::2]
                    p2_mask = invalid_action_masks[1::2]

                    m_action, _, _, _ = macro_agent.get_action_and_value(
                        p2_obs,
                        device=device,
                    )

                    unit_mask = envs.get_source_unit_mask()[1::2]
                    m_action = m_action.cpu() * unit_mask
                    m_action = m_action.view(1, 16, 16).numpy()

                    micro_action = None

                    for num_agent, agent in enumerate(agents):

                        action_mask = torch.tensor(np.where(
                            m_action == num_agent,
                            1,
                            0,
                        ))
                        action_mask = torch.unsqueeze(action_mask, -1)
                        action_mask = action_mask.to(device)
                        micro_obs = torch.cat([p2_obs, action_mask], -1)

                        action, _, _, _, _ = agent.get_action_and_value(
                            micro_obs,
                            envs=envs,
                            invalid_action_masks=p2_mask,
                            device=device,
                        )

                        action_mask = action_mask.view(1, 256, 1)
                        action = action * action_mask.expand(action.size())

                        if micro_action is None:
                            micro_action = action
                        else:
                            micro_action += action

                    p1_action, _, _, _, _ = agent2.get_action_and_value(
                        p1_obs,
                        envs=envs,
                        invalid_action_masks=p1_mask,
                        device=device,
                    )
                    action = torch.zeros(
                        (
                            args.num_envs,
                            p1_action.shape[1],
                            p1_action.shape[2],
                        ))
                    action[::2] = p1_action
                    action[1::2] = micro_action

            try:
                next_obs, rs, ds, infos = envs.step(
                    action.cpu().numpy().reshape(envs.num_envs, -1))
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise

            for idx, info in enumerate(infos):
                if "episode" in info.keys():
                    result = info["microrts_stats"]["WinLossRewardFunction"]
                    if args.ai:
                        print("against", args.ai, result)

                        if result > 0:
                            wins += 1
                        elif result < 0:
                            losses += 1
                        else:
                            ties += 1

                        print(
                            f"W:{wins} - L:{losses} D:{ties} - "
                            f"Total: {wins + losses + ties}"
                            )

                        if (wins + losses + ties) >= args.num_games:
                            envs.close()
                            exit(0)

                    else:
                        if idx % 2 == 0:
                            print(f"player{idx % 2}", result)

                            if result > 0:
                                wins += 1
                            elif result < 0:
                                losses += 1
                            else:
                                ties += 1

                            if (wins + losses + ties) >= args.num_games:
                                envs.close()
                                exit(0)

    envs.close()
