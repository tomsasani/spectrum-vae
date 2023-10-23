import simulation
import generator
import global_vars

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root_dist = np.array([0.25, 0.25, 0.25, 0.25])

sim = simulation.simulate_exp
N = 5_000

# first, initialize generator object
gen = generator.Generator(
    sim,
    [global_vars.NUM_HAPLOTYPES // 2],
    ["kappa"],
    np.random.randint(1, 2**32),
)

train = gen.simulate_batch(N, root_dist, [], treat_as_real=True, plot=True)

idx = np.random.randint(N)

# f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 2, figsize=(12, 6))
if global_vars.NUM_CHANNELS == 1:
    f, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(train[idx, :, :, 0], ax=ax, cbar=True)
    f.tight_layout()
    f.savefig('test.png', dpi=200)
else:
    f, axarr = plt.subplots(global_vars.NUM_CHANNELS, figsize=(12, 6))
    for channel_i in range(global_vars.NUM_CHANNELS):
        sns.heatmap(train[idx, :, :, channel_i], ax=axarr[channel_i], cbar=True)
        #sns.heatmap(pos[idx, :, :, channel_i], ax=axarr[channel_i, 1], cbar=True)
    f.tight_layout()
    f.savefig('test.png', dpi=200)

DATADIR = "data"

np.savez(f"{DATADIR}/data.npz", train=train)

# simulate a bunch of training examples
neg = gen.simulate_batch(N, root_dist, [], treat_as_real=True, plot=True)
pos = gen.simulate_batch(N, root_dist, [3], plot=False)

np.savez(f"{DATADIR}/labeled.npz", neg=neg, pos=pos)


