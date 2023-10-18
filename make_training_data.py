from PIL import Image
import simulation 
import generator 
import global_vars

import numpy as np

root_dist = np.array([0.25, 0.25, 0.25, 0.25])

sim = simulation.simulate_exp

N = 50_000 

# first, initialize generator object
gen = generator.Generator(
    sim,
    [global_vars.NUM_HAPLOTYPES // 2],
    np.random.randint(1, 2**32),
)

# simulate a bunch of training examples
train = gen.simulate_batch(N, root_dist, mutator_threshold=0.)
test = gen.simulate_batch(N // 4, root_dist, mutator_threshold=0.)

DATADIR = "data"

np.savez(f"{DATADIR}/data.npz", train=train, test=test)

# simulate a bunch of training examples
# neg = gen.simulate_batch(N, root_dist, mutator_threshold=0.)
# pos = gen.simulate_batch(N, root_dist, mutator_threshold=1)

# np.savez(f"{DATADIR}/labeled.npz", neg=neg, pos=pos)
