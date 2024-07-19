# Adapted from http://proceedings.mlr.press/v97/han19a/han19a.pdf

import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import (
    VecEnvWrapper,
    VecMonitor,
    VecVideoRecorder,
)
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from helpers import (
    get_advantages_from_returns,
    get_returns_from_advantages,
    optimise,
)
from macro_agent import MacroAgent
from micro_agent import MicroAgent
from ppo_utils import parse_args, EpochData, get_rewards


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos[:])
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i])
                * np.concatenate(
                    (infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()),
                    axis=None,
                )
            ]
            self.ts[i] += 1
            if dones[i]:
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                raw_discount_returns = np.array(
                    self.raw_discount_rewards[i]
                ).sum(0)
                raw_discount_names = [
                    "discounted_" + str(rf) for rf in self.rfs
                ] + ["discounted"]
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(dict(zip(
                    raw_discount_names,
                    raw_discount_returns,
                )))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


if __name__ == "__main__":
    args = parse_args()

    print(f"Save frequency: {args.save_frequency}")

    # TRY NOT TO MODIFY: setup the environment
    tmstamp = int(time.time())
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{tmstamp}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|"for key, value in vars(args).items()])
        )
    )

    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    print(f"Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
        + [microrts_ai.randomBiasedAI for _ in range(min(
            args.num_bot_envs,
            2,
        ))]
        + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))]
        + [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
        map_paths=[args.train_maps[0]],
        reward_weight=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(
            envs,
            f"videos/{experiment_name}",
            record_video_trigger=lambda x: x % 100000 == 0, video_length=2000,
        )
    assert isinstance(
        envs.action_space,
        MultiDiscrete,
    ), "only MultiDiscrete action space is supported"

    macro_agent = MacroAgent(envs).to(device)
    macro_optimizer = optim.Adam(
        macro_agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    move_agent = MicroAgent(envs).to(device)
    move_optimizer = optim.Adam(
        move_agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    attack_agent = MicroAgent(envs).to(device)
    attack_optimizer = optim.Adam(
        attack_agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    harvest_agent = MicroAgent(envs).to(device)
    harvest_optimizer = optim.Adam(
        harvest_agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    produce_agent = MicroAgent(envs).to(device)
    produce_optimizer = optim.Adam(
        produce_agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    agents = [move_agent, attack_agent, harvest_agent, produce_agent]

    if args.anneal_lr:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
        def lr(f):
            return f * args.learning_rate

    # ALGO Logic: Storage for epoch data
    h, w, c = envs.observation_space.shape
    mapsize = h * w
    action_space_shape = (mapsize, len(envs.action_plane_space.nvec))
    invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

    # Macro Agent
    macro_epoch_data = EpochData(
        args.num_steps,
        args.num_envs,
        h, w, c,
        1,
        device,
    )

    move_epoch_data = EpochData(
        args.num_steps,
        args.num_envs,
        h, w, (c + 1),
        len(envs.action_plane_space.nvec),
        device,
        has_iam=True,
        iam_shape=invalid_action_shape,
        reward_weight=np.array([10.0, 0, 0, 0, 0, 0]),
    )

    attack_epoch_data = EpochData(
        args.num_steps,
        args.num_envs,
        h, w, (c + 1),
        len(envs.action_plane_space.nvec),
        device,
        has_iam=True,
        iam_shape=invalid_action_shape,
        reward_weight=np.array([10.0, 0, 0, 0, 1.0, 0]),
    )

    harvest_epoch_data = EpochData(
        args.num_steps,
        args.num_envs,
        h, w, (c + 1),
        len(envs.action_plane_space.nvec),
        device,
        has_iam=True,
        iam_shape=invalid_action_shape,
        reward_weight=np.array([10.0, 1.0, 0, 0, 0, 0]),
    )

    produce_epoch_data = EpochData(
        args.num_steps,
        args.num_envs,
        h, w, (c + 1),
        len(envs.action_plane_space.nvec),
        device,
        has_iam=True,
        iam_shape=invalid_action_shape,
        reward_weight=np.array([10.0, 0, 1.0, 0.2, 0, 4.0]),
    )

    epoch_datas = [
        move_epoch_data,
        attack_epoch_data,
        harvest_epoch_data,
        produce_epoch_data,
    ]

    optimisers = [
        move_optimizer,
        attack_optimizer,
        harvest_optimizer,
        produce_optimizer,
    ]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done`
    # are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # CRASH AND RESUME LOGIC:
    starting_update = 1

    print("Macro model's state_dict:")
    for param_tensor in macro_agent.state_dict():
        print(
            param_tensor,
            "\t",
            macro_agent.state_dict()[param_tensor].size(),
        )
    total_params = sum(
        [param.nelement() for param in macro_agent.parameters()]
    )
    print("Model's total parameters:", total_params)
    print("..............................................................")

    print("Move model's state_dict:")
    for param_tensor in move_agent.state_dict():
        print(
            param_tensor,
            "\t",
            move_agent.state_dict()[param_tensor].size(),
        )
    total_params = sum(
        [param.nelement() for param in move_agent.parameters()]
    )
    print("Model's total parameters:", total_params)
    print("..............................................................")

    print("Attack model's state_dict:")
    for param_tensor in attack_agent.state_dict():
        print(
            param_tensor,
            "\t",
            attack_agent.state_dict()[param_tensor].size(),
        )
    total_params = sum(
        [param.nelement() for param in attack_agent.parameters()]
    )
    print("Model's total parameters:", total_params)
    print("..............................................................")

    print("Harvest model's state_dict:")
    for param_tensor in harvest_agent.state_dict():
        print(
            param_tensor,
            "\t",
            harvest_agent.state_dict()[param_tensor].size(),
        )
    total_params = sum(
        [param.nelement() for param in harvest_agent.parameters()]
    )
    print("Model's total parameters:", total_params)
    print("..............................................................")

    print("Produce model's state_dict:")
    for param_tensor in produce_agent.state_dict():
        print(
            param_tensor,
            "\t",
            produce_agent.state_dict()[param_tensor].size(),
        )
    total_params = sum(
        [param.nelement() for param in produce_agent.parameters()]
    )
    print("Model's total parameters:", total_params)
    print("..............................................................")

    for update in range(starting_update, args.num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lrnow = lr(frac)
            macro_optimizer.param_groups[0]["lr"] = lrnow
            move_optimizer.param_groups[0]["lr"] = lrnow
            attack_optimizer.param_groups[0]["lr"] = lrnow
            harvest_optimizer.param_groups[0]["lr"] = lrnow
            produce_optimizer.param_groups[0]["lr"] = lrnow

        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            # envs.render()
            global_step += 1 * args.num_envs
            macro_epoch_data.obs[step] = next_obs
            macro_epoch_data.dones[step] = next_done
            macro_rs = 0
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                m_action, logproba, _, vs = macro_agent.get_action_and_value(
                    next_obs,
                    device=device,
                )

                macro_epoch_data.values[step] = vs.flatten()

            macro_epoch_data.actions[step] = m_action
            macro_epoch_data.logprobs[step] = logproba

            m_action = m_action * envs.get_source_unit_mask()
            m_action = m_action.view(args.num_envs, h, w).cpu().numpy()

            for num_agent, agent_zip in enumerate(zip(agents, epoch_datas)):
                agent, data = agent_zip

                action_mask = torch.tensor(np.where(
                    m_action == num_agent,
                    1,
                    0,
                ))
                action_mask = torch.unsqueeze(action_mask, -1)

                micro_obs = torch.cat([next_obs, action_mask], -1)

                data.obs[step] = micro_obs
                data.dones[step] = next_done

                with torch.no_grad():
                    data.invalid_action_masks[step] = torch.tensor(
                        envs.get_action_mask()
                    ).to(device)

                    # action -> 24, 256, 7
                    action, logproba, _, _, vs = agent.get_action_and_value(
                        micro_obs,
                        envs=envs,
                        invalid_action_masks=data.invalid_action_masks[step],
                        device=device,
                    )

                    data.values[step] = vs.flatten()

                data.actions[step] = action
                data.logprobs[step] = logproba

                action_mask = action_mask.view(args.num_envs, h * w, 1)
                action = action * action_mask.expand(action.size())

                try:
                    next_obs, _, ds, infos = envs.step(
                        action.cpu().numpy().reshape(envs.num_envs, -1)
                    )

                    next_obs = torch.Tensor(next_obs).to(device)
                    micro_rs = get_rewards(infos, data.reward_weight)

                    macro_rs += micro_rs
                except Exception as e:
                    e.printStackTrace()
                    raise

                data.rewards[step] = torch.Tensor(micro_rs).to(device)
                next_done = torch.Tensor(ds).to(device)

            macro_epoch_data.rewards[step] = torch.Tensor(macro_rs).to(device)

            for info in infos:
                if "episode" in info.keys():
                    print(
                        f"global_step={global_step}, "
                        f"episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return",
                        info["episode"]["r"], global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        info["episode"]["l"],
                        global_step,
                    )
                    for key in info["microrts_stats"]:
                        writer.add_scalar(
                            f"charts/episodic_return/{key}",
                            info["microrts_stats"][key], global_step,
                        )
                    break

        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            m_action, _, _, last_macro_v = macro_agent.get_action_and_value(
                next_obs,
                device=device,
            )

            last_macro_v = last_macro_v.reshape(1, -1)

            if args.gae:
                get_returns_from_advantages(
                    args.num_steps,
                    next_done,
                    args.gamma,
                    args.gae_lambda,
                    last_macro_v,
                    macro_epoch_data,
                    device,
                )
            else:
                get_advantages_from_returns(
                    args.num_steps,
                    next_done,
                    args.gamma,
                    last_macro_v,
                    macro_epoch_data,
                    device,
                )

            m_action = m_action * envs.get_source_unit_mask()
            m_action = m_action.view(args.num_envs, h, w).cpu().numpy()

            for num_agent, agent_zip in enumerate(zip(agents, epoch_datas)):
                agent, data = agent_zip

                action_mask = torch.tensor(np.where(
                    m_action == num_agent,
                    1,
                    0,
                ))
                action_mask = torch.unsqueeze(action_mask, -1)
                micro_obs = torch.cat([next_obs, action_mask], -1)
                last_micro_value = agent.get_value(micro_obs).reshape(1, -1)

                if num_agent < len(agents) - 1:
                    d_index = num_agent + 1
                    d_step = args.num_steps - 1
                    micro_next_done = epoch_datas[d_index].dones[d_step]
                else:
                    micro_next_done = next_done

                if args.gae:
                    get_returns_from_advantages(
                        args.num_steps,
                        micro_next_done,
                        args.gamma,
                        args.gae_lambda,
                        last_micro_value,
                        data,
                        device,
                    )
                else:
                    get_advantages_from_returns(
                        args.num_steps,
                        micro_next_done,
                        args.gamma,
                        last_micro_value,
                        data,
                        device,
                    )

        # flatten the batch

        macro_epoch_data.flatten(h, w, c, 1)

        optimise(
            macro_epoch_data,
            macro_optimizer,
            args.update_epochs,
            args.batch_size,
            args.minibatch_size,
            args.norm_adv,
            macro_agent,
            "macro",
            envs,
            device,
            args.clip_coef,
            args.clip_vloss,
            args.ent_coef,
            args.vf_coef,
            update,
            global_step,
            writer,
            args.kle_stop,
            args.kle_rollback,
            args.max_grad_norm,
        )

        for num_agent, agent_zip in enumerate(zip(
            agents, epoch_datas, optimisers
        )):
            agent, data, optimiser = agent_zip
            data.flatten(
                h, w, c + 1,
                len(envs.action_plane_space.nvec),
                has_iam=True,
                iam_shape=invalid_action_shape,
            )

            optimise(
                data,
                optimiser,
                args.update_epochs,
                args.batch_size,
                args.minibatch_size,
                args.norm_adv,
                agent,
                num_agent,
                envs,
                device,
                args.clip_coef,
                args.clip_vloss,
                args.ent_coef,
                args.vf_coef,
                update,
                global_step,
                writer,
                args.kle_stop,
                args.kle_rollback,
                args.max_grad_norm,
                has_iam=True,
            )

        if (update - 1) % args.save_frequency == 0:
            if not os.path.exists(f"models/{experiment_name}"):
                os.makedirs(f"models/{experiment_name}")

            torch.save(
                macro_agent.state_dict(),
                f"models/{experiment_name}/macro_agent.pt",
            )

            torch.save(
                move_agent.state_dict(),
                f"models/{experiment_name}/move_agent.pt",
            )

            torch.save(
                attack_agent.state_dict(),
                f"models/{experiment_name}/attack_agent.pt",
            )

            torch.save(
                produce_agent.state_dict(),
                f"models/{experiment_name}/produce_agent.pt",
            )

            torch.save(
                harvest_agent.state_dict(),
                f"models/{experiment_name}/harvest_agent.pt",
            )

            # global step saves

            os.makedirs(f"models/{experiment_name}/{global_step}")

            torch.save(
                macro_agent.state_dict(),
                f"models/{experiment_name}/{global_step}/macro.pt",
            )

            torch.save(
                move_agent.state_dict(),
                f"models/{experiment_name}/{global_step}/move.pt",
            )

            torch.save(
                attack_agent.state_dict(),
                f"models/{experiment_name}/{global_step}/attack.pt",
            )

            torch.save(
                produce_agent.state_dict(),
                f"models/{experiment_name}/{global_step}/produce.pt",
            )

            torch.save(
                harvest_agent.state_dict(),
                f"models/{experiment_name}/{global_step}/harvest.pt",
            )

        writer.add_scalar(
            "charts/sps",
            int(global_step / (time.time() - start_time)),
            global_step,
        )
        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    writer.close()
