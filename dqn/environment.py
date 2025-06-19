# -*- coding: utf-8 -*
import gymnasium as gym
import numpy as np
import torchvision.transforms as transforms


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))


def wrap_env(env: gym.Env) -> gym.Env:
    """
    Wrap Atari environment with transforms:
    - Resize to 64x64
    - Transform to grayscale
    - Stack frames (temporal)
    - Clip reward
    """
    xform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.Lambda(lambda x: np.array(x)),
        ]
    )

    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: xform(obs),
        gym.spaces.Box(low=0, high=255, shape=(64, 64), dtype=np.uint8),
    )

    env = gym.wrappers.FrameStackObservation(env, 4)  # Stack four frames
    env = ClipRewardEnv(env)  # Clip the reward to {+1, 0, -1} by its sign.

    return env
