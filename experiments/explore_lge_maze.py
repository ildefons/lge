import os

import gym_continuous_maze
import numpy as np
from stable_baselines3 import SAC
from toolbox.maze_grid import compute_coverage
import matplotlib.pyplot as plt
from lge import LatentGoExplore

NUM_TIMESTEPS = 100_000
NUM_RUN = 1

for run_idx in range(NUM_RUN):
    model = LatentGoExplore(
        SAC,
        "ContinuousMaze-v0",
        module_type="inverse",
        latent_size=16,
        distance_threshold=1.0,
        lighten_dist_coef=1.0,
        p=0.05,
        model_kwargs=dict(buffer_size=NUM_TIMESTEPS),
        verbose=1,
    )
    model.explore(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations) / (24 * 24) * 100
    coverage = np.expand_dims(coverage, 0)

    # filename = "results/lge_maze.npy"
    # if os.path.exists(filename):
    #     previous_coverage = np.load(filename)
    #     coverage = np.concatenate((previous_coverage, coverage))
    # np.save(filename, coverage)

    #plots: ILDE: I get it from the notebook
    bins = np.floor(observations)  # Divide the space into 1 x 1 bins.
    unique, bin_uids = np.unique(bins, axis=0, return_inverse=True)  # Each bin has its own UID

    explored_uid_so_far = []
    coverage = np.zeros(NUM_TIMESTEPS)
    for t in range(NUM_TIMESTEPS):
        bin_uid = bin_uids[t]
        if not bin_uid in explored_uid_so_far:
            explored_uid_so_far.append(bin_uid)
        coverage[t] = len(explored_uid_so_far) / (24 * 24)  # there are 24 x 24 reachable bins

    plt.xlabel("timesteps")
    plt.ylabel("Space coverage rate")
    plt.plot(coverage)
    plt.show()

    # print(1)
    # print(2)