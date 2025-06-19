# -*- coding: utf-8 -*
import yaml
import dacite
import argparse

import ale_py
import gymnasium as gym


# Local
from .dqn import train
from .state import State
from .config import Config
from .model import QNetwork
from .buffer import ReplayBuffer
from .environment import wrap_env


def train_cli():
    gym.register_envs(ale_py)

    parser = argparse.ArgumentParser(description="DQL Training script")
    parser.add_argument("-c", "--config", default="cfg/space_invaders.yaml")

    args = parser.parse_args()
    with open(args.config) as fd:
        config_dict = yaml.safe_load(fd)
        cfg = dacite.from_dict(data_class=Config, data=dict(config_dict))

    envs = gym.vector.SyncVectorEnv(
        [
            lambda: wrap_env(gym.make(cfg.env_id, render_mode="rgb_array"))
            for _ in range(cfg.n_envs)  # Create environments
        ]
    )

    eval_envs = gym.vector.SyncVectorEnv(
        [
            lambda: wrap_env(gym.make(cfg.env_id, render_mode="rgb_array"))
            for _ in range(cfg.n_envs)  # Create environments
        ]
    )

    state = State(cfg)
    rb = ReplayBuffer(cfg.replay_buffer_capacity)
    q_network = QNetwork(envs.single_action_space.n).to(cfg.device)

    # Load checkpoint if any
    state.checkpoint_load(cfg, q_network, rb)

    # Hard training ))
    try:
        train(envs, eval_envs, cfg, state, rb, q_network)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        state.checkpoint_save(cfg, q_network, rb, force=True)


def eval_cli():
    pass
