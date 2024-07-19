import numpy as np
import torch
import torch.nn as nn


def get_returns_from_advantages(
    num_steps,
    next_done,
    gamma,
    gae_lambda,
    last_value,
    data,
    device,
):
    advantages = torch.zeros_like(data.rewards).to(device)
    lastgaelam = 0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = last_value
        else:
            nextnonterminal = 1.0 - data.dones[t + 1]
            nextvalues = data.values[t + 1]

        # delta
        delta = data.rewards[t] + gamma * nextvalues * nextnonterminal
        delta -= data.values[t]

        advantages[t] = delta
        advantages[t] += gamma * gae_lambda * nextnonterminal * lastgaelam

        lastgaelam = advantages[t]

    returns = advantages + data.values

    data.advantages = advantages
    data.returns = returns


def get_advantages_from_returns(
    num_steps,
    next_done,
    gamma,
    last_value,
    data,
    device,
):
    returns = torch.zeros_like(data.rewards).to(device)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            next_return = last_value
        else:
            nextnonterminal = 1.0 - data.dones[t + 1]
            next_return = returns[t + 1]

        returns[t] = data.rewards[t] + gamma * nextnonterminal * next_return

    advantages = returns - data.values

    data.advantages = advantages
    data.returns = returns


def optimise(
    data,
    optimizer,
    update_epochs,
    batch_size,
    minibatch_size,
    norm_adv,
    agent,
    tag,
    envs,
    device,
    clip_coef,
    clip_vloss,
    ent_coef,
    vf_coef,
    update,
    global_step,
    writer,
    kle_stop,
    kle_rollback,
    max_grad_norm,
    has_iam=False,
):

    inds = np.arange(batch_size)

    for i_epoch_pi in range(update_epochs):
        np.random.shuffle(inds)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = data.b_advantages[minibatch_ind]

            if norm_adv:
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / (mb_advantages.std() + 1e-8)

            if has_iam:
                results = agent.get_action_and_value(
                    data.b_obs[minibatch_ind],
                    data.b_actions.long()[minibatch_ind],
                    data.b_invalid_action_masks[minibatch_ind],
                    envs,
                    device,
                )
                _, newlogproba, entropy, _, new_values = results
            else:
                results = agent.get_action_and_value(
                    data.b_obs[minibatch_ind],
                    data.b_actions.long()[minibatch_ind],
                    device,
                )
                _, newlogproba, entropy, new_values = results

            ratio = (newlogproba - data.b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (data.b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio,
                1 - clip_coef,
                1 + clip_coef,
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            new_values = new_values.view(-1)
            if clip_vloss:
                v_loss_unclipped = (
                    new_values - data.b_returns[minibatch_ind]
                ) ** 2
                v_clipped = data.b_values[minibatch_ind] + torch.clamp(
                    new_values - data.b_values[minibatch_ind],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (
                    v_clipped - data.b_returns[minibatch_ind]
                ) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * (
                    (new_values - data.b_returns[minibatch_ind]) ** 2
                )

            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            # plotting

    writer.add_scalar(
        f"{tag}/charts/learning_rate",
        optimizer.param_groups[0]["lr"],
        global_step,
    )
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar(
        f"{tag}/losses/value_loss",
        v_loss.detach().item(),
        global_step,
    )
    writer.add_scalar(
        f"{tag}/losses/policy_loss",
        pg_loss.detach().item(),
        global_step,
    )
    writer.add_scalar(
        f"{tag}/losses/entropy",
        entropy.detach().mean().item(),
        global_step,
    )
    writer.add_scalar(
        f"{tag}/losses/approx_kl",
        approx_kl.detach().item(),
        global_step,
    )
    if kle_stop or kle_rollback:
        writer.add_scalar(
            f"{tag}/debug/pg_stop_iter",
            i_epoch_pi,
            global_step,
        )
