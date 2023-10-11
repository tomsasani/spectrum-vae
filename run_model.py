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

### BUILD MODEL

LATENT_DIM = 8
model = autoencoder.CAE(latent_dim=LATENT_DIM, kernel_size=5)
model.build_graph((
    1,
    global_vars.NUM_HAPLOTYPES,
    global_vars.NUM_SNPS,
    global_vars.NUM_CHANNELS,
))
model.encoder.summary()
model.decoder.summary()

optimizer = tf.keras.optimizers.legacy.Adam()
loss_object = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)


BATCH_SIZE = 32

root_dist = np.array([0.25, 0.25, 0.25, 0.25])

sim = simulation.simulate_exp

# first, initialize generator object
gen = generator.Generator(
    sim,
    global_vars.NUM_HAPLOTYPES // 2,
    np.random.randint(1, 2**32),
)

# train_dataset = tf.data.Dataset.from_tensor_slices(
#     (train_data, np.zeros(NUM_TRAIN))).batch(BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_tensor_slices(
#     (test_data, np.zeros(NUM_TEST))).batch(BATCH_SIZE)

train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/images/",
    batch_size=BATCH_SIZE,
    shuffle=False,
    image_size=(global_vars.NUM_HAPLOTYPES, global_vars.NUM_SNPS),
    color_mode="grayscale",
    validation_split=0.2,
    subset="both",
    labels=None,
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x: (normalization_layer(x)))
test_ds = test_ds.map(lambda x: (normalization_layer(x)))

EPOCHS = 5

res = []

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

    for images in train_ds:
        train_step(images, images)
    for images in test_ds:
        test_step(images, images)

    print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            #f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            #f'Test Accuracy: {test_accuracy.result() * 100}'
    )
    res.append({"Training loss": train_loss.result(),
                "Validation loss": test_loss.result(),
                "Epoch": epoch + 1})

res_df = pd.DataFrame(res)

f, ax = plt.subplots()
ax.plot(res_df["Training loss"], label="Training Loss")
ax.plot(res_df["Validation loss"], label="Validation Loss")
ax.legend()
f.savefig("loss.png", dpi=200)

# create some new simulated data. we'll use these data to figure out the reconstruction
# error of the model on unseen data.

# simulate a bunch of training examples
normal_data = gen.simulate_batch(100, root_dist, mutator_threshold=0)
mutator_data = gen.simulate_batch(100, root_dist, mutator_threshold=1)

encoded_data = model.encoder(normal_data).numpy()
decoded_data = model.decoder(encoded_data).numpy()

f, ax = plt.subplots()
n = 2
to_plot = np.arange(n)
f, axarr = plt.subplots(global_vars.NUM_CHANNELS, n * 2, figsize=(8, n))
for channel_i in np.arange(global_vars.NUM_CHANNELS):
    for idx, plot_i in enumerate(to_plot):
        sns.heatmap(normal_data[plot_i, :, :, channel_i], ax=axarr[idx * 2], cbar=True)
        sns.heatmap(decoded_data[plot_i, :, :, channel_i], ax=axarr[(idx * 2) + 1], cbar=True)
# for idx in range(n):
#     axarr[idx * 2].set_title("R")
#     axarr[(idx * 2) + 1].set_title("G")
f.tight_layout()
f.savefig("real_vs_decoded.png", dpi=200)

normal_reconstructions = model.predict(normal_data)
mutator_reconstructions = model.predict(mutator_data)
normal_loss = tf.reduce_mean(losses.mse(normal_reconstructions, normal_data), axis=(1, 2)).numpy()
mutator_loss = tf.reduce_mean(losses.mse(mutator_reconstructions, mutator_data), axis=(1, 2)).numpy()

max_loss = np.max([np.max(normal_loss), np.max(mutator_loss)])
min_loss = np.min([np.min(normal_loss), np.min(mutator_loss)])

bins = np.linspace(min_loss, max_loss, num=50)
f, ax = plt.subplots()
ax.hist(normal_loss, bins=bins, alpha=0.5, label="Normal")
ax.hist(mutator_loss, bins=bins, alpha=0.5, label="Mutator")
ax.legend()
ax.set_xlabel("Loss")
ax.set_ylabel("No of examples")
f.savefig("recon_loss.png", dpi=200)