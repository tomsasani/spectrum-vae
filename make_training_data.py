from PIL import Image
import simulation 
import generator 
import global_vars

import numpy as np

root_dist = np.array([0.25, 0.25, 0.25, 0.25])

sim = simulation.simulate_exp

# first, initialize generator object
gen = generator.Generator(
    sim,
    global_vars.NUM_HAPLOTYPES // 2,
    np.random.randint(1, 2**32),
)

N = 50_000

# simulate a bunch of training examples
data = gen.simulate_batch(N, root_dist, mutator_threshold=0)

DATADIR = "data/images"

for i, img in enumerate(data):
    fh = f"{DATADIR}/{i}.png"
    arr = img[:, :, 0].astype('uint8')
    arr *= 255
    image = Image.fromarray(arr)
    image.save(fh)