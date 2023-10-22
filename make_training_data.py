import simulation
import generator
import global_vars

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root_dist = np.array([0.25, 0.25, 0.25, 0.25])

sim = simulation.simulate_exp
N = 10_000

# first, initialize generator object
gen = generator.Generator(
    sim,
    [global_vars.NUM_HAPLOTYPES // 2],
    np.random.randint(1, 2**32),
)

# simulate a bunch of training examples
train = gen.simulate_batch(N, root_dist, mutator_threshold=0., plot=True)
#test = gen.simulate_batch(N // 4, root_dist, mutator_threshold=0.)

# train = train.reshape((
#     N * global_vars.NUM_HAPLOTYPES,
#     global_vars.NUM_SNPS,
#     global_vars.NUM_CHANNELS,
# ))
# test = test.reshape((
#     (N // 4) * global_vars.NUM_HAPLOTYPES,
#     global_vars.NUM_SNPS,
#     global_vars.NUM_CHANNELS,
# ))
idx = np.random.randint(N)

# f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 2, figsize=(12, 6))
if global_vars.NUM_CHANNELS == 1:
    f, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(train[idx, :, :, 0], ax=ax, cbar=True)
    f.tight_layout()
    f.savefig('test.png', dpi=200)
else:
    f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 2, figsize=(12, 6))
    for channel_i in range(global_vars.NUM_CHANNELS):
        sns.heatmap(train[idx, :, :, channel_i], ax=axarr[channel_i, 0], cbar=True)
        sns.heatmap(train[idx, :, :, channel_i], ax=axarr[channel_i, 1], cbar=True)
    f.tight_layout()
    f.savefig('test.png', dpi=200)

DATADIR = "data"

np.savez(f"{DATADIR}/data.npz", train=train)

test_normal = gen.simulate_batch(N, root_dist, mutator_threshold=0)
test_mutator = gen.simulate_batch(N, root_dist, mutator_threshold=1)

# test_normal = test_normal.reshape((
#     N * global_vars.NUM_HAPLOTYPES,
#     global_vars.NUM_SNPS,
#     global_vars.NUM_CHANNELS,
# ))
# test_mutator = test_mutator.reshape((
#     N * global_vars.NUM_HAPLOTYPES,
#     global_vars.NUM_SNPS,
#     global_vars.NUM_CHANNELS,
# ))

np.savez(f"{DATADIR}/labeled.npz", neg=test_normal, pos=test_mutator)
