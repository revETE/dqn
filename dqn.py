# %% Imports
from reprlib import Repr
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import random
import numpy as np

import ale_py
import gymnasium as gym

from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

# Local
from model import QNetwork
from buffer import ReplayBuffer
from environment import wrap_env

# Paper reported setup
# - minibatch size: 32
# - replay buffer size: 1_000_000
# - exploration at thr begining: 1.0
# - exploration at the end: 0.1
# - exploration frames: 1_000_000
# - training trames: 10_000_000


@dataclass
class Config:
    device: str = "cuda"

    env_id: str = "ALE/SpaceInvaders-v5"
    exp_name: str = "default"

    n_envs: int = 32
    n_timesteps: int = 350_000  # Number of training timesteps
    n_eval_timesteps: int = 1_000  # Number of evaluation timesteps

    exploration_timesteps: int = 35_000  # Timesteps till exploration end
    exploration_begin: float = 1.00  # Exploration ration at the begining
    exploration_end: float = 0.1  # Exploration ration at the end

    gamma_discount: float = 0.99  # Value function discounting factor

    replay_buffer_capacity: int = 1_000_000  # Replay buffer capacity
    replay_buffer_sampling: int = 1024  # Replay buffer sample size

    target_update_frequency: int = 1000  # Target network update fraquency
    evaluate_model_frequency: int = 1000

    optimizer_lr: float = 1e-3

    checkpoint_frequency: int = 1000  # Global checkpoint frequency
    checkpoint_buffer_frequency: int = 10_000  # Replay buffer checkpoint frequency
    checkpoint_path: str = "run"
    checkpoint_model_name: str = "model"
    checkpoint_state_name: str = "state"
    checkpoint_buffer_name: str = "buffer"

    tensorboard_path: str = "run/tb"  # Tensorboard runs storage path

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._run_name = f"{self.env_id.replace('/', '-')}__{self.exp_name}__{int(time.time())}"

    def run_name(self) -> str:
        return self._run_name

    def ckp_name(self) -> str:
        return f"{self.env_id.replace('/', '-')}__{self.exp_name}__{int(time.time())}"


@dataclass
class State:
    total_length: np.ndarray
    total_reward = np.ndarray
    total_games = np.ndarray

    reward_ema: float = 0.0
    length_ema: float = 0.0
    ema_alpha: float = 0.05

    timestep: int = 1
    epoch: int = 0

    def __init__(self, n_envs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total_length = np.zeros((n_envs,))
        self.total_reward = np.zeros((n_envs,))
        self.total_games = np.zeros((n_envs,))

    def update(self, timestep: int, rewards: np.ndarray, terminated: np.ndarray) -> None:
        self.timestep = timestep
        self.total_length += 1
        self.total_reward += rewards

        if np.any(terminated):
            # Update length
            for length in self.total_length[terminated]:
                self.length_ema = length * self.ema_alpha + self.length_ema * (1.0 - self.ema_alpha)

            # Update EMA
            for reward in self.total_reward[terminated]:
                self.reward_ema = reward * self.ema_alpha + self.reward_ema * (1.0 - self.ema_alpha)

            self.total_length[terminated] = 0  # Reset lengths
            self.total_reward[terminated] = 0  # Reset rewards
            self.total_games[terminated] += 1  # Update number of played games

    def next_epoch(self):
        self.epoch += 1

    def report_tb(self, writer, timestep: int, epsilon: float, loss: float):
        if timestep % 10 == 0:
            writer.add_scalar("episode/length_ema", self.length_ema, timestep)
            writer.add_scalar("episode/reward_ema", self.reward_ema, timestep)
            writer.add_scalar("episode/epsilon", epsilon, timestep)
            writer.add_scalar("games/loss", loss, timestep)
            writer.add_scalar("games/total", self.total_games.sum(), timestep)

    def report_pbar(self, pbar, timestep: int, epsilon: float, loss: float, rb: ReplayBuffer):
        pbar.set_description(
            f"step={timestep} epoch={self.epoch} "
            + f"length={self.length_ema:.2f} reward={self.reward_ema:.2f} "
            + f"games={self.total_games.sum():.0f} rb_used={rb.used()}/{rb.capacity()} "
            + f"epsilon={epsilon:.3f} loss={loss:.3f}"
        )

    def checkpoint_save(self, cfg: Config, q_network: QNetwork, rb: ReplayBuffer, force=False):
        # Model checkpoint (storing every snapshot)
        model_current = f"{cfg.checkpoint_path}/{cfg.ckp_name()}__{cfg.checkpoint_model_name}.pth"
        model_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_model_name}.pth"
        torch.save(q_network.state_dict(), model_current)
        os.symlink(os.path.abspath(model_current), f"{cfg.checkpoint_path}/tmp.link")
        os.rename(f"{cfg.checkpoint_path}/tmp.link", model_latest)

        # Current state checkpoint (storing only latest, overriding previous)
        state_current = f"{cfg.checkpoint_path}/{cfg.run_name()}__{cfg.checkpoint_state_name}.npz"
        state_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_state_name}.npz"
        self.save(state_current)
        os.symlink(os.path.abspath(state_current), f"{cfg.checkpoint_path}/tmp.link")
        os.rename(f"{cfg.checkpoint_path}/tmp.link", state_latest)

        # Replay Buffer checkpoint (storing only latest, overriding previous)
        if self.timestep % cfg.checkpoint_buffer_frequency == 0 or force:
            rb_current = f"{cfg.checkpoint_path}/{cfg.run_name()}__{cfg.checkpoint_buffer_name}.npz"
            rb_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_buffer_name}.npz"
            rb.save(rb_current)
            os.symlink(os.path.abspath(rb_current), f"{cfg.checkpoint_path}/tmp.link")
            os.rename(f"{cfg.checkpoint_path}/tmp.link", rb_latest)

    def checkpoint_load(self, cfg: Config, q_network: QNetwork, rb: ReplayBuffer) -> None:
        model_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_model_name}.pth"
        state_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_state_name}.npz"
        buffer_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_buffer_name}.npz"

        if os.path.exists(model_latest):
            q_network.load_state_dict(torch.load(model_latest, weights_only=True))
            print(f"Model checkpoint loaded from {model_latest}")

        if os.path.exists(state_latest):
            self.load(state_latest)
            print(f"State checkpoint loaded from {state_latest}")

        if os.path.exists(buffer_latest):
            rb.load(buffer_latest)
            print(f"Buffer checkpoint loaded from {buffer_latest}")

    def save(self, path: str) -> None:
        np.savez(
            path,
            total_length=self.total_length,
            total_reward=self.total_reward,
            total_games=self.total_games,
            reward_ema=self.reward_ema,
            length_ema=self.length_ema,
            ema_alpha=self.ema_alpha,
            timestep=self.timestep,
            epoch=self.epoch,
        )

    def load(self, path: str) -> None:
        npzfile = np.load(path)

        self.total_length = npzfile["total_length"]
        self.total_reward = npzfile["total_reward"]
        self.total_games = npzfile["total_games"]

        self.reward_ema = npzfile["reward_ema"]
        self.length_ema = npzfile["length_ema"]
        self.ema_alpha = npzfile["ema_alpha"]
        self.timestep = npzfile["timestep"]
        self.epoch = npzfile["epoch"]


def train(
    envs,
    cfg: Config,
    state: State,
    rb: ReplayBuffer,
    q_network: QNetwork,
    writer: SummaryWriter,
):
    # Create Traget Network and load weights from Q-Network
    target_network = QNetwork(envs.single_action_space.n).to(cfg.device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.AdamW(q_network.parameters(), lr=cfg.optimizer_lr)

    observations, info = envs.reset()
    for timestep in (pbar := tqdm(range(state.timestep, cfg.n_timesteps + 1))):
        # Epsilon greedy policy with linear schedule
        slope = (cfg.exploration_end - cfg.exploration_begin) / cfg.exploration_timesteps
        epsilon = np.maximum(slope * timestep + cfg.exploration_begin, cfg.exploration_end)

        if random.random() < epsilon:
            actions = envs.action_space.sample()  # shape=(n_envs,)
        else:
            with torch.no_grad():
                x = torch.Tensor(observations).to(cfg.device)
                q_values = q_network(x)  # shape=(n_envs, n_actions)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()  # shape=(n_envs,)

        # Perform action and receive observation and reward
        (
            observations_next,  # shape=(n_envs, n_frames, d_rows, d_cols)
            rewards,  # shape=(n_envs,)
            terminated,  # shape=(n_envs,)
            _,
            info,
        ) = envs.step(actions)
        state.update(timestep, rewards, terminated)  # Update metrics and state

        # Store observations in replay buffer
        rb.add(observations, observations_next, actions, rewards, terminated)
        observations = observations_next

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
        q_values = q_network(train_observations)

        # Calculate mse loss
        y = q_values.gather(1, train_actions.view(cfg.replay_buffer_sampling, 1))
        loss = F.mse_loss(y_target, y.view(cfg.replay_buffer_sampling))

        # Gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        if timestep % cfg.target_update_frequency == 0:
            state.next_epoch()

            for target_params, q_params in zip(target_network.parameters(), q_network.parameters()):
                target_params.data.copy_(q_params.data)

        if timestep % cfg.evaluate_model_frequency == 0:
            eval(cfg, q_network, writer, timestep)
            q_network.train()

        #
        # Model and Replay Buffer checkpoint
        #
        if timestep % cfg.checkpoint_frequency == 0:
            state.checkpoint_save(cfg, q_network, rb)

        state.report_tb(writer, timestep, epsilon, loss.cpu().detach().numpy().item())
        state.report_pbar(pbar, timestep, epsilon, loss.cpu().detach().numpy().item(), rb)


def eval(cfg: Config, q_network: QNetwork, writer: SummaryWriter, global_timestep: int):
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: wrap_env(gym.make(cfg.env_id, render_mode="rgb_array"))
            for _ in range(cfg.n_envs)  # Create environments
        ]
    )

    total_length = np.zeros((cfg.n_envs,))
    total_reward = np.zeros((cfg.n_envs,))
    total_games = np.zeros((cfg.n_envs,))

    rewards_list = []
    lengths_list = []

    q_network.eval()
    observations, info = envs.reset()
    for timestep in (pbar := tqdm(range(cfg.n_eval_timesteps))):
        with torch.no_grad():
            x = torch.Tensor(observations).to(cfg.device)
            q_values = q_network(x)  # shape=(n_envs, n_actions)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()  # shape=(n_envs,)

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

            # Update EMA
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

    writer.add_scalar("episode/eval_length_avg", np.mean(lengths_list), global_timestep)
    writer.add_scalar("episode/eval_reward_avg", np.mean(rewards_list), global_timestep)
    writer.add_scalar("episode/eval_length_std", np.std(lengths_list), global_timestep)
    writer.add_scalar("episode/eval_reward_std", np.std(rewards_list), global_timestep)
    writer.add_scalar("games/eval_total", total_games.sum(), global_timestep)


def main():
    gym.register_envs(ale_py)

    # TODO: Load config from json
    cfg = Config()

    envs = gym.vector.SyncVectorEnv(
        [
            lambda: wrap_env(gym.make(cfg.env_id, render_mode="rgb_array"))
            for _ in range(cfg.n_envs)  # Create environments
        ]
    )

    state = State(cfg.n_envs)
    rb = ReplayBuffer(cfg.replay_buffer_capacity)
    q_network = QNetwork(envs.single_action_space.n).to(cfg.device)

    # Load checkpoint if any
    state.checkpoint_load(cfg, q_network, rb)

    # TB Summary writer
    writer = SummaryWriter(f"{cfg.tensorboard_path}/{cfg.run_name()}")

    # Hard training ))
    try:
        train(envs, cfg, state, rb, q_network, writer)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        state.checkpoint_save(cfg, q_network, rb, force=True)
        writer.close()


if __name__ == "__main__":
    main()
