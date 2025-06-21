# -*- coding: utf-8 -*
import random
import imageio
import numpy as np

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

# Local
from .model import QNetwork
from .state import State
from .video import make_image
from .config import Config
from .buffer import ReplayBuffer


def train(envs, eval_envs, cfg: Config, state: State, rb: ReplayBuffer, q_network, optimizer):
    # Video recording
    if cfg.record_train_video:
        size = int(np.ceil(np.sqrt(cfg.n_envs)).item())
        output_path = f"{cfg.checkpoint_path}/{cfg.ckp_name()}_train.mp4"
        writer = imageio.get_writer(output_path, format="FFMPEG", mode="I", fps=60)

    # Create Traget Network and load weights from Q-Network
    target_network = QNetwork(envs.single_action_space.n).to(cfg.device)
    target_network.load_state_dict(q_network.state_dict())

    if state.timestep > 1:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.optimizer_lr,
            div_factor=2,
            last_epoch=state.timestep,
            total_steps=cfg.n_timesteps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.optimizer_lr,
            div_factor=2,
            total_steps=cfg.n_timesteps,
        )

    observations, info = envs.reset()

    # Collect initial data
    while state.timestep < cfg.replay_buffer_sampling:
        actions = envs.action_space.sample()
        # Sync ENV API: Perform actions and Receive observation and reward
        (
            observations_next,  # shape=(n_envs, n_frames, d_rows, d_cols)
            rewards,  # shape=(n_envs,)
            terminated,  # shape=(n_envs,)
            _,
            info,
        ) = envs.step(actions)
        state.update(rewards, terminated)  # Update metrics and state

        # Store observations in replay buffer
        rb.add(observations, observations_next, actions, rewards, terminated)
        observations = observations_next

    for step in range(state.timestep, cfg.n_timesteps + 1):
        # Video recording
        if cfg.record_train_video:
            writer.append_data(make_image(observations, size, size))

        # Epsilon greedy policy with linear schedule
        slope = (cfg.exploration_end - cfg.exploration_begin) / cfg.exploration_timesteps
        epsilon = np.maximum(slope * step + cfg.exploration_begin, cfg.exploration_end)

        if random.random() < epsilon:
            actions = envs.action_space.sample()  # shape=(n_envs,)
        else:
            with torch.no_grad():
                x = torch.Tensor(observations).to(cfg.device)
                q_values = q_network(x)  # shape=(n_envs, n_actions)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()  # shape=(n_envs,)

        # Async ENV API: Perform actions
        envs.send(actions)

        # Sample minibatch from replay buffer
        (
            train_observations,  # shape=(n_samples, n_frames, d_rows, d_cols)
            train_observations_next,  # shape=(n_samples, n_frames, d_rows, d_cols)
            train_actions,  # shape=(n_samples,)
            train_rewards,  # shape=(n_samples,)
            train_terminated,  # shape=(n_samples,)
        ) = rb.sample(cfg.replay_buffer_sampling)

        # TD target estimation
        with torch.no_grad():
            # Convert sample to torch tensor
            train_rewards = torch.tensor(train_rewards).to(cfg.device)
            train_terminated = torch.tensor(train_terminated).to(cfg.device)
            train_observations_next = torch.tensor(train_observations_next).to(cfg.device)

            # Predict
            q_values_next, _ = target_network(train_observations_next).max(dim=1)

            # y_target = train_rewards (if episode termitates)
            # y_target = train_rewards + cfg.gamma_discount * q_values_next (otherwise)
            y_target = train_rewards + cfg.gamma_discount * q_values_next * (1 - train_terminated)

        # Forward pass and loss calculation
        train_actions = torch.tensor(train_actions).to(dtype=torch.int64).to(cfg.device)
        train_observations = torch.Tensor(train_observations).to(cfg.device)

        # Forward pass
        q_values = q_network(train_observations)  # shape=(n_samples, n_actions)

        # Calculate mse loss
        y = q_values.gather(1, train_actions.view(cfg.replay_buffer_sampling, 1))
        loss = F.mse_loss(y_target, y.view(cfg.replay_buffer_sampling))

        # Gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Async ENV API: Receive observation and reward
        (
            observations_next,  # shape=(n_envs, n_frames, d_rows, d_cols)
            rewards,  # shape=(n_envs,)
            terminated,  # shape=(n_envs,)
            _,
            info,
        ) = envs.recv()
        state.update(rewards, terminated)  # Update metrics and state

        # Store observations in replay buffer
        rb.add(observations, observations_next, actions, rewards, terminated)
        observations = observations_next

        # Update target network
        if step % cfg.target_update_frequency == 0:
            state.next_epoch()  # Update metrics and state

            for target_params, q_params in zip(target_network.parameters(), q_network.parameters()):
                target_params.data.copy_(q_params.data)

        # Model and Replay Buffer checkpoint
        if step % cfg.checkpoint_frequency == 0:
            state.checkpoint_save(cfg, q_network, optimizer, rb)

            # TODO: Record train video via state
            if cfg.record_train_video:
                writer.close()
                output_path = f"{cfg.checkpoint_path}/{cfg.ckp_name()}_train.mp4"
                writer = imageio.get_writer(output_path, format="FFMPEG", mode="I", fps=60)

        # Pause training and evaluate
        if step % cfg.evaluate_model_frequency == 0:
            eval(eval_envs, cfg, state, q_network)
            q_network.train()

        state.report(
            rb,
            epsilon,
            loss.cpu().detach().numpy().item(),
            optimizer.param_groups[0]["lr"],
        )


def eval(envs, cfg: Config, state: State, q_network: QNetwork):
    # Video recording
    if cfg.record_eval_video:
        size = int(np.ceil(np.sqrt(cfg.n_envs)).item())
        output_path = f"{cfg.checkpoint_path}/{cfg.ckp_name()}_eval.mp4"
        writer = imageio.get_writer(output_path, format="FFMPEG", mode="I", fps=60)

    # Metrics
    total_length = np.zeros((envs.num_envs,))
    total_reward = np.zeros((envs.num_envs,))
    total_games = np.zeros((envs.num_envs,))

    rewards_list = []
    lengths_list = []

    q_network.eval()
    observations, info = envs.reset()
    for timestep in (pbar := tqdm(range(cfg.n_eval_timesteps))):
        with torch.no_grad():
            x = torch.Tensor(observations).to(cfg.device)
            q_values = q_network(x)  # shape=(n_envs, n_actions)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()  # shape=(n_envs,)

        # Video recording
        if cfg.record_eval_video:
            writer.append_data(make_image(observations, size, size))

        # Perform action and receive observation and reward
        (
            observations_next,  # shape=(n_envs, n_frames, d_rows, d_cols)
            rewards,  # shape=(n_envs,)
            terminated,  # shape=(n_envs,)
            _,
            info,
        ) = envs.step(actions)
        observations = observations_next

        total_length += 1
        total_reward += rewards

        if np.any(terminated):
            # Update length
            for length in total_length[terminated]:
                lengths_list.append(length)

            # Update reward
            for reward in total_reward[terminated]:
                rewards_list.append(reward)

            total_length[terminated] = 0  # Reset lengths
            total_reward[terminated] = 0  # Reset rewards
            total_games[terminated] += 1  # Update number of played games

        if len(lengths_list) > 0 and len(rewards_list) > 0:
            stats = (
                f"length_avg={np.mean(lengths_list):.2f} reward_avg={np.mean(rewards_list):.2f} "
                + f"length_std={np.std(lengths_list):.2f} reward_std={np.std(rewards_list):.2f} "
            )
        else:
            stats = ""

        pbar.set_description(f"step={timestep} " + stats + f"games={total_games.sum():.0f}")

    state.report_eval(
        np.mean(lengths_list),
        np.mean(rewards_list),
        np.std(lengths_list),
        np.std(rewards_list),
        total_games.sum(),
    )

    if cfg.record_eval_video:
        writer.close()
