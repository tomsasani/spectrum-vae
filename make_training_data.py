import simulation
import generator
import global_vars

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sim = simulation.simulate_exp
N = 1_000

ONE_HOT = True

# first, initialize generator object
gen = generator.Generator(
    sim,
    [global_vars.NUM_HAPLOTYPES // 2],
    [],
    np.random.randint(1, 2**32),
)

train_normal = gen.simulate_batch(
    N,
    [],
    kappa=2.0,
    treat_as_real=True,
    one_hot=ONE_HOT,
    plot=True,
)
labels_normal = np.zeros(N * global_vars.NUM_HAPLOTYPES)
#labels_normal = np.zeros(N)

train_weird = gen.simulate_batch(
    N // 10,
    [],
    kappa=10.0,
    treat_as_real=True,
    one_hot=ONE_HOT,
    plot=False,
)
labels_weird = np.ones((N // 10) * global_vars.NUM_HAPLOTYPES)
#labels_weird = np.ones((N // 10))

X_train = np.concatenate((train_normal, train_weird), axis=0)
y_train = np.concatenate((labels_normal, labels_weird))

print (X_train.shape, y_train.shape)

idx = np.random.randint(N)

# f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 2, figsize=(12, 6))
if global_vars.NUM_CHANNELS == 1:
    f, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(train_normal[idx, :, :, 0, 0], ax=ax, cbar=False)
    f.tight_layout()
    f.savefig("test.png", dpi=200)
else:
    f, axarr = plt.subplots(global_vars.NUM_CHANNELS, figsize=(8, 16))
    for channel_i in range(global_vars.NUM_CHANNELS):
        sns.heatmap(
            train_normal[0, :, :, 0, channel_i],
            ax=axarr[channel_i],
            cbar=True,
            vmin=-1,
            vmax=1,
        )
        axarr[channel_i].set_xticks([])
    f.tight_layout()
    f.savefig("test.png", dpi=200)

DATADIR = "data"

np.savez(f"{DATADIR}/data.npz", train=X_train, labels=y_train)
