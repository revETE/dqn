# -*- coding: utf-8 -*
import yaml
import dacite
import argparse

import torch
import ale_py
import gymnasium as gym

from ale_py.vector_env import AtariVectorEnv

# Local
from .dqn import train
from .state import State
from .config import Config
from .model import QNetwork
from .buffer import ReplayBuffer


def train_cli():
    gym.register_envs(ale_py)

    parser = argparse.ArgumentParser(description="DQL Training script")
    parser.add_argument("-c", "--config", default="cfg/space_invaders.yaml")

    args = parser.parse_args()
    with open(args.config) as fd:
        config_dict = yaml.safe_load(fd)
        cfg = dacite.from_dict(data_class=Config, data=dict(config_dict))

    # Create a vector environment
    envs = AtariVectorEnv(game=cfg.env_id, num_envs=cfg.n_envs, thread_affinity_offset=1)
    eval_envs = AtariVectorEnv(game=cfg.env_id, num_envs=1)

    state = State(cfg)
    rb = ReplayBuffer(cfg.replay_buffer_capacity)
    q_network = QNetwork(envs.single_action_space.n).to(cfg.device)
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=cfg.optimizer_lr)

    # Load checkpoint if any
    state.checkpoint_load(cfg, q_network, optimizer, rb)

    # Hard training ))
    try:
        train(envs, eval_envs, cfg, state, rb, q_network, optimizer)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        state.checkpoint_save(cfg, q_network, optimizer, rb, force=True)
        state.close()


def eval_cli():
    pass
