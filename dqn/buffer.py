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
        self.observation = np.ndarray((self._capacity, 4, 84, 84), dtype=np.uint8)
        self.observation_next = np.ndarray((self._capacity, 4, 84, 84), dtype=np.uint8)

    def add(self, observations, observations_next, actions, rewards, terminated):
        n_samples, n_frames, d_rows, d_cols = observations.shape

        if self._write_position + n_samples >= self._capacity:
            head = self._capacity - self._write_position
            tail = n_samples - head

            rb_ix_from = self._write_position
            rb_ix_till = self._write_position + head

            ob_ix_from = 0
            ob_ix_till = head

            # Assign
            self.action[rb_ix_from:rb_ix_till] = actions[ob_ix_from:ob_ix_till]
            self.reward[rb_ix_from:rb_ix_till] = rewards[ob_ix_from:ob_ix_till]
            self.terminated[rb_ix_from:rb_ix_till] = terminated[ob_ix_from:ob_ix_till]
            self.observation[rb_ix_from:rb_ix_till] = observations[ob_ix_from:ob_ix_till]
            self.observation_next[rb_ix_from:rb_ix_till] = observations_next[ob_ix_from:ob_ix_till]

            self._write_position = 0
            self._overwrite = True

            rb_ix_from = self._write_position
            rb_ix_till = self._write_position + tail

            ob_ix_from = head
            ob_ix_till = n_samples

        else:
            rb_ix_from = self._write_position
            rb_ix_till = self._write_position + n_samples

            ob_ix_from = 0
            ob_ix_till = n_samples

        # Assign
        self.action[rb_ix_from:rb_ix_till] = actions[ob_ix_from:ob_ix_till]
        self.reward[rb_ix_from:rb_ix_till] = rewards[ob_ix_from:ob_ix_till]
        self.terminated[rb_ix_from:rb_ix_till] = terminated[ob_ix_from:ob_ix_till]
        self.observation[rb_ix_from:rb_ix_till] = observations[ob_ix_from:ob_ix_till]
        self.observation_next[rb_ix_from:rb_ix_till] = observations_next[ob_ix_from:ob_ix_till]

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
