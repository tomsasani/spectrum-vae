from PIL import Image
import simulation 
import generator 
import global_vars

import numpy as np

root_dist = np.array([0.25, 0.25, 0.25, 0.25])

sim = simulation.simulate_exp

N = 1_000 

# first, initialize generator object
gen = generator.Generator(
    sim,
    [50],
    np.random.randint(1, 2**32),
)

# simulate a bunch of training examples
train = gen.simulate_batch(N, root_dist, mutator_threshold=0)
test = gen.simulate_batch(N // 4, root_dist, mutator_threshold=0)

DATADIR = "data"

np.savez(f"{DATADIR}/data.npz", train=train, test=test)