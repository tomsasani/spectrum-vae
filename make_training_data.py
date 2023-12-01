import simulation
import generator
import global_vars

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sim = simulation.simulate_exp
N = 10_000

normal_root_dist = np.array([0.25] * 4)

ONE_HOT = False

# first, initialize generator object
gen = generator.Generator(
    sim,
    [global_vars.NUM_HAPLOTYPES // 2],
    ["mu"],
    np.random.randint(1, 2**32),
)

train_normal = gen.simulate_batch(
    N,
    normal_root_dist,
    [1e-8],
    effect_size=1.0,
    treat_as_real=True,
    one_hot=ONE_HOT,
    plot=True,
)
labels_normal = np.zeros(N)# * global_vars.NUM_HAPLOTYPES)

train_weird = gen.simulate_batch(
    N // 1_000,
    normal_root_dist,
    [1e-8],
    effect_size=1.0,
    treat_as_real=True,
    one_hot=ONE_HOT,
    plot=False,
)
labels_weird = np.ones((N // 1_000))

X_train = np.concatenate((train_normal, train_weird), axis=0)
y_train = np.concatenate((labels_normal, labels_weird))

print (X_train.shape, y_train.shape)

idx = np.random.randint(N)

# f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 2, figsize=(12, 6))
if global_vars.NUM_CHANNELS == 1:
    f, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(train_normal[idx, :, :], ax=ax, cbar=False)
    f.tight_layout()
    f.savefig("test.png", dpi=200)
else:
    f, axarr = plt.subplots(global_vars.NUM_CHANNELS, figsize=(6, 12))
    for channel_i in range(global_vars.NUM_CHANNELS):
        sns.heatmap(
            train_normal[0, :, :, channel_i],
            ax=axarr[channel_i],
            cbar=False,
            vmin=-1,
            vmax=1,
        )
    f.tight_layout()
    f.savefig("train_normal.png", dpi=200)

    f, axarr = plt.subplots(global_vars.NUM_CHANNELS, figsize=(6, 12))
    for channel_i in range(global_vars.NUM_CHANNELS):
        sns.heatmap(
            train_weird[0, :, :, channel_i],
            ax=axarr[channel_i],
            cbar=False,
            vmin=-1,
            vmax=1,
        )
    f.tight_layout()
    f.savefig("train_weird.png", dpi=200)

DATADIR = "data"

np.savez(f"{DATADIR}/data.npz", train_data=X_train, train_labels=y_train)
