# -*- coding: utf-8 -*
import os
import torch
import numpy as np

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

# Local
from .model import QNetwork
from .config import Config
from .buffer import ReplayBuffer


class State:
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        self._timestep = 1
        self._epoch = 0
        self._resumed = False

        super().__init__(*args, **kwargs)
        self.total_length = np.zeros((cfg.n_envs,))
        self.total_reward = np.zeros((cfg.n_envs,))
        self.total_games = np.zeros((cfg.n_envs,))

        self.reward_ema = 0.0
        self.length_ema = 0.0
        self.ema_alpha = 0.05

        # Store config
        self.cfg = cfg

        # Create tqdm progress bar
        self.pbar = tqdm(total=cfg.n_timesteps)

        # Create TB Summary writer
        self.writer = SummaryWriter(f"{cfg.tensorboard_path}/{cfg.run_name()}")

    def close(self):
        self.pbar.close()
        self.writer.close()

    @property
    def epoch(self):
        return self._epoch

    def next_epoch(self):
        self._epoch += 1

    @property
    def resumed(self):
        return self._resumed

    @property
    def timestep(self):
        return self._timestep + 0  # Return copy, not reference

    def next_timestep(self):
        self._timestep += 1

    def update(self, rewards: np.ndarray, terminated: np.ndarray) -> None:
        self.next_timestep()
        self.total_length += 1
        self.total_reward += rewards

        # Upadte progress bar
        self.pbar.n = self.timestep
        self.pbar.refresh()

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

    def report(self, rb: ReplayBuffer, epsilon: float, loss: float, lr: float):
        self._report_tb(self.writer, self.timestep, epsilon, loss, lr)
        self._report_pbar(self.pbar, self.timestep, epsilon, loss, rb, lr)

    def _report_tb(self, writer, timestep: int, epsilon: float, loss: float, lr: float):
        if timestep % 10 == 0:
            writer.add_scalar("train/length_ema", self.length_ema, timestep)
            writer.add_scalar("train/reward_ema", self.reward_ema, timestep)
            writer.add_scalar("train/epsilon", epsilon, timestep)
            writer.add_scalar("train/lr", lr, timestep)
            writer.add_scalar("train/loss", loss, timestep)
            writer.add_scalar("train/games", self.total_games.sum(), timestep)

    def _report_pbar(self, pbar, timestep: int, epsilon: float, loss: float, rb, lr: float):
        pbar.set_description(
            f"step={timestep} epoch={self.epoch} "
            + f"length={self.length_ema:.2f} reward={self.reward_ema:.2f} "
            + f"games={self.total_games.sum():.0f} used={rb.used()}/{rb.capacity()} "
            + f"epsilon={epsilon:.3f} loss={loss:.3f} lr={lr:.5f}"
        )

    def report_eval(self, length_avg, reward_avg, length_std, reward_std, total):
        self.writer.add_scalar("eval/length_avg", length_avg, self.timestep)
        self.writer.add_scalar("eval/reward_avg", reward_avg, self.timestep)
        self.writer.add_scalar("eval/length_std", length_std, self.timestep)
        self.writer.add_scalar("eval/reward_std", reward_std, self.timestep)
        self.writer.add_scalar("eval/games", total, self.timestep)

    def checkpoint_save(self, cfg: Config, q_network, optimizer, rb: ReplayBuffer, force=False):
        # Model checkpoint (storing every snapshot)
        model_current = f"{cfg.checkpoint_path}/{cfg.ckp_name()}__{cfg.checkpoint_model_name}.pth"
        model_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_model_name}.pth"
        torch.save(
            {
                "q_network_state_dict": q_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_current,
        )
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

    def checkpoint_load(self, cfg: Config, q_network: QNetwork, optimizer, rb: ReplayBuffer):
        model_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_model_name}.pth"
        if len(cfg.resume_model_path) > 0:
            model_latest = cfg.resume_model_path

        state_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_state_name}.npz"
        buffer_latest = f"{cfg.checkpoint_path}/latest__{cfg.checkpoint_buffer_name}.npz"

        if os.path.exists(model_latest):
            checkpoint = torch.load(model_latest, weights_only=True)
            q_network.load_state_dict(checkpoint["q_network_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Model checkpoint loaded from {model_latest}")
            self._resumed = True

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
        self._timestep = npzfile["timestep"]
        self._epoch = npzfile["epoch"]

        # Update pbar with initial progress
        self.pbar.update(self.timestep)
