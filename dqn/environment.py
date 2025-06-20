# -*- coding: utf-8 -*
import gymnasium as gym
import numpy as np
import torchvision.transforms as transforms


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward) -> float:
        return np.sign(float(reward))


def wrap_env(env: gym.Env) -> gym.Env:
    resize = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((75, 55)),
            transforms.Lambda(lambda x: np.array(x)),
        ]
    )

    # Crop
    env = gym.wrappers.TransformObservation(
        env,
        lambda observation: observation[45:195, 25:135],
        gym.spaces.Box(low=0, high=255, shape=(150, 110), dtype=np.uint8),
    )

    # Resize
    env = gym.wrappers.TransformObservation(
        env,
        lambda observation: resize(observation),
        gym.spaces.Box(low=0, high=255, shape=(75, 55), dtype=np.uint8),
    )

    env = gym.wrappers.FrameStackObservation(env, 4)  # Stack four frames
    env = ClipRewardEnv(env)  # Clip the reward to {+1, 0, -1} by its sign.

    return env
