import generator
import simulation
import global_vars
import autoencoder

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import losses
import tensorflow as tf
import pandas as pd 
import time

NUM_TRAIN_EXAMPLES = 50_000
NUM_TEST_EXAMPLES = NUM_TRAIN_EXAMPLES // 2

root_dist = np.array([0.25, 0.25, 0.25, 0.25])

sim = simulation.simulate_exp

# first, initialize generator object
gen = generator.Generator(
    sim,
    global_vars.NUM_HAPLOTYPES // 2,
    np.random.randint(1, 2**32),
)

# simulate a bunch of training examples
train_data = gen.simulate_batch(NUM_TRAIN_EXAMPLES, root_dist, mutator_threshold=0.)
test_data = gen.simulate_batch(NUM_TEST_EXAMPLES, root_dist, mutator_threshold=0.)

f, ax = plt.subplots()
n = 2
to_plot = np.arange(n)
f, axarr = plt.subplots(global_vars.NUM_CHANNELS, n * 2, figsize=(24, n * 2))
for channel_i in np.arange(global_vars.NUM_CHANNELS):
    for idx, plot_i in enumerate(to_plot):
        sns.heatmap(train_data[plot_i, :, :, channel_i], ax=axarr[idx * 2], cbar=True)
        sns.heatmap(test_data[plot_i, :, :, channel_i], ax=axarr[(idx * 2) + 1], cbar=True)
for idx in range(n):
    axarr[idx * 2].set_title("Train")
    axarr[(idx * 2) + 1].set_title("Test")
f.tight_layout()
f.savefig("training.png")

### RUN MODEL 
LATENT_DIM = 64
model = autoencoder.CAE(latent_dim=LATENT_DIM, kernel_size=5)
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_data, train_data, 
          epochs=20, 
          batch_size=32,
          validation_data=(test_data, test_data),
          shuffle=True)
model.encoder.summary()
model.decoder.summary()

f, ax = plt.subplots()
ax.plot(history.history["loss"], label="Training Loss")
ax.plot(history.history["val_loss"], label="Validation Loss")
ax.legend()
f.savefig("loss.png", dpi=200)

# create some new simulated data. we'll use these data to figure out the reconstruction
# error of the model on unseen data.

# simulate a bunch of training examples
normal_data = gen.simulate_batch(NUM_TEST_EXAMPLES, root_dist, mutator_threshold=0.)
mutator_data = gen.simulate_batch(NUM_TEST_EXAMPLES, root_dist, mutator_threshold=0.1)

encoded_data = model.encoder(normal_data).numpy()
decoded_data = model.decoder(encoded_data).numpy()

f, ax = plt.subplots()
n = 2
to_plot = np.arange(n)
f, axarr = plt.subplots(global_vars.NUM_CHANNELS, n * 2, figsize=(24, n * 2))
for channel_i in np.arange(global_vars.NUM_CHANNELS):
    for idx, plot_i in enumerate(to_plot):
        sns.heatmap(normal_data[plot_i, :, :, channel_i], ax=axarr[idx * 2], cbar=True)
        sns.heatmap(decoded_data[plot_i, :, :, channel_i], ax=axarr[(idx * 2) + 1], cbar=True)
for idx in range(n):
    axarr[idx * 2].set_title("R")
    axarr[(idx * 2) + 1].set_title("G")
f.tight_layout()
f.savefig("real_vs_decoded.png")

normal_reconstructions = model.predict(normal_data)
mutator_reconstructions = model.predict(mutator_data)
normal_loss = tf.reduce_mean(losses.mse(normal_reconstructions, normal_data), axis=(1, 2)).numpy()
mutator_loss = tf.reduce_mean(losses.mse(mutator_reconstructions, mutator_data), axis=(1, 2)).numpy()

max_loss = np.max([np.max(normal_loss), np.max(mutator_loss)])
bins = np.linspace(0, max_loss, num=50)
f, ax = plt.subplots()
ax.hist(normal_loss, bins=bins, alpha=0.25, label="Normal")
ax.hist(mutator_loss, bins=bins, alpha=0.25, label="Mutator")
ax.legend()
ax.set_xlabel("Loss")
ax.set_ylabel("No of examples")
f.savefig("recon_loss.png", dpi=200)