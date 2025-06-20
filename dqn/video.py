# -*- coding: utf-8 -*
import imageio
import numpy as np


def make_image(observations, n_rows, n_cols):
    n_samples, n_frames, d_rows, d_cols = observations.shape
    image = np.zeros((d_rows * n_rows, d_cols * n_cols), observations.dtype)

    for row in range(n_rows):
        for col in range(n_cols):
            if observations.shape[0] > n_rows * row + col:
                image[
                    d_rows * row : d_rows * (row + 1),
                    d_cols * col : d_cols * (col + 1),
                ] = observations[n_rows * row + col, 0]

    return image


def make_sample_video(envs, output="output.mp4", steps=1000):
    with imageio.get_writer(output, format="FFMPEG", mode="I", fps=60) as writer:
        observations, info = envs.reset()
        writer.append_data(make_image(observations, 4, 4))

        for _ in range(steps):
            actions = envs.action_space.sample()
            observations, _, _, _, _ = envs.step(actions)

            writer.append_data(make_image(observations, 4, 4))
