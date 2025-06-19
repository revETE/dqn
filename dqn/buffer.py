# -*- coding: utf-8 -*
import os
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self._capacity = capacity
        self.reset()

        self.action = np.ndarray((self._capacity,), dtype=np.uint8)
        self.reward = np.ndarray((self._capacity,), dtype=np.float32)
        self.terminated = np.ndarray((self._capacity,), dtype=np.uint8)
        self.observation = np.ndarray((self._capacity, 4, 64, 64), dtype=np.uint8)
        self.observation_next = np.ndarray((self._capacity, 4, 64, 64), dtype=np.uint8)

    def add(self, observations, observations_next, actions, rewards, terminated):
        n_samples, n_frames, d_rows, d_cols = observations.shape

        if self._write_position + n_samples >= self._capacity:
            self._write_position = 0
            self._overwrite = True

        ix_from = self._write_position
        ix_till = self._write_position + n_samples

        # Assign
        self.action[ix_from:ix_till] = actions
        self.reward[ix_from:ix_till] = rewards
        self.terminated[ix_from:ix_till] = terminated
        self.observation[ix_from:ix_till] = observations
        self.observation_next[ix_from:ix_till] = observations_next

        self._write_position += n_samples

    def sample(self, size):
        if self._overwrite:
            ix = np.random.randint(0, self._capacity, size=size)
        else:
            ix = np.random.randint(0, self._write_position, size=size)

        actions = self.action[ix]
        rewards = self.reward[ix]
        terminated = self.terminated[ix]
        observation = self.observation[ix, :, :, :]
        observation_next = self.observation_next[ix, :, :, :]

        return observation, observation_next, actions, rewards, terminated

    def reset(self):
        self._write_position = 0
        self._overwrite = False

    def capacity(self):
        return self._capacity

    def used(self):
        if self._overwrite:
            return self._capacity
        else:
            return self._write_position

    def free(self):
        return self.capacity() - self.used()

    def save(self, path: str):
        np.save(f"{path}.action", self.action, allow_pickle=False)
        np.save(f"{path}.reward", self.reward, allow_pickle=False)
        np.save(f"{path}.terminated", self.terminated, allow_pickle=False)
        np.save(f"{path}.observation", self.observation, allow_pickle=False)
        np.save(f"{path}.observation_next", self.observation_next, allow_pickle=False)

        np.savez(
            path,
            write_position=self._write_position,
            overwrite=self._overwrite,
        )

    def load(self, path: str):
        # State
        npzfile = np.load(path)
        self._write_position = npzfile["write_position"]
        self._overwrite = npzfile["overwrite"]

        self._capacity = self.observation.shape[0]

        # Data
        if os.path.islink(path):
            path = os.readlink(path)

        self.action = np.load(f"{path}.action.npy")
        self.reward = np.load(f"{path}.reward.npy")
        self.terminated = np.load(f"{path}.terminated.npy")
        self.observation = np.load(f"{path}.observation.npy")
        self.observation_next = np.load(f"{path}.observation_next.npy")
