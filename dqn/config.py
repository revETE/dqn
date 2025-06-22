# -*- coding: utf-8 -*
import os
import time
import torch
import numpy as np

from dataclasses import dataclass

from .buffer import ReplayBuffer
from .model import QNetwork

# Paper reported setup
# - minibatch size: 32
# - replay buffer size: 1_000_000
# - exploration at thr begining: 1.0
# - exploration at the end: 0.1
# - exploration frames: 1_000_000
# - training trames: 10_000_000


@dataclass
class Config:
    device: str = "cpu"

    env_id: str = "SpaceInvaders-v4"
    exp_name: str = "default"

    n_envs: int = 25
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
    scheduler_pct_start: float = 0.3  # OneCycleLR pct_start
    scheduler_div_factor: float = 2  # OneCycleLR div_factor

    checkpoint_frequency: int = 1000  # Global checkpoint frequency
    checkpoint_buffer_frequency: int = 10_000  # Replay buffer checkpoint frequency
    checkpoint_path: str = "run"
    checkpoint_model_name: str = "model"
    checkpoint_state_name: str = "state"
    checkpoint_buffer_name: str = "buffer"

    tensorboard_path: str = "run/tb"  # Tensorboard runs storage path

    record_train_video: bool = False
    record_eval_video: bool = True

    def run_name(self) -> str:
        if not hasattr(self, "_run_name"):
            self._run_name = f"{self.env_id.replace('/', '-')}__{self.exp_name}__{int(time.time())}"
        return self._run_name

    def ckp_name(self) -> str:
        return f"{self.env_id.replace('/', '-')}__{self.exp_name}__{int(time.time())}"
