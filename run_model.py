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
import wandb
import argparse



### BUILD MODEL
def build_model(
    initial_filters: int = 8,
    conv_layers: int = 2,
    latent_dimensions: int = 8,
    fc_layers: int = 0,
    fc_layer_size: int = 32,
    dropout: float = 0.5,
):

    model = autoencoder.CAE(
        initial_filters=initial_filters,
        conv_layers=conv_layers,
        latent_dimensions=latent_dimensions,
        fc_layers=fc_layers,
        fc_layer_size=fc_layer_size,
        dropout=dropout,
    )
    model.build_graph((
        1,
        global_vars.NUM_HAPLOTYPES,
        global_vars.NUM_SNPS,
        global_vars.NUM_CHANNELS,
    ))
    model.encoder.summary()
    model.decoder.summary()

    return model

def build_optimizer(learning_rate: float = 1e-3):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    return optimizer

def build_data(batch_size: int = 32):
    with np.load("data/data.npz") as data:
        train_data = data["train"]
        test_data = data["test"]
    train_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)
    return train_ds, test_ds


def train_step(images, labels, model, optimizer, loss_object, train_loss):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


def test_step(images, labels, model, loss_object, test_loss):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)


def train_model_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_ds, test_ds = build_data(batch_size = config.batch_size)
        model = build_model(
            initial_filters=config.initial_filters,
            conv_layers=config.conv_layers,
            latent_dimensions=config.latent_dimensions,
            fc_layers=config.fc_layers,
            fc_layer_size=config.fc_layer_size,
            dropout=config.dropout,
        )
        optimizer = build_optimizer(learning_rate=config.learning_rate)

        loss_object = tf.keras.losses.MeanSquaredError()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        for epoch in range(config.epochs):
            train_loss.reset_states()
            test_loss.reset_states()

            for images in train_ds:
                train_step(images, images, model, optimizer, loss_object, train_loss)
            for images in test_ds:
                test_step(images, images, model, loss_object, test_loss)

            validation_loss = test_loss.result()

            wandb.log({"validation_loss": validation_loss, "epoch": epoch})

def main(args):

    ### define the sweep
    sweep_config = {"method": "random"}

    metric = {'name': 'validation_loss', 'goal': 'minimize'}

    sweep_config['metric'] = metric

    # set up parameters over which to sweep
    parameters_dict = {
        'optimizer': {
            'values': ['adam']
        },
        'initial_filters': {'values': [4, 8, 16, 32]},
        'conv_layers': {'values': [2, 3]},
        'latent_dimensions': {'values': [8, 16, 32, 64]},
        'fc_layers': {'values': [0, 1, 2]},
        'fc_layer_size': {
            'values': [32, 64]
        },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
        },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 1e-3
        },
        'batch_size': {
            'values': [32, 64],
        },
        'epochs': {'value': 10}
    }

    sweep_config['parameters'] = parameters_dict

    with np.load("data/data.npz") as data:
        train_data = data["train"]
        test_data = data["test"]

    f, ax = plt.subplots()
    n = 2
    to_plot = np.arange(n)
    f, axarr = plt.subplots(global_vars.NUM_CHANNELS, n * 2, figsize=(12, n * 6))
    for channel_i in np.arange(global_vars.NUM_CHANNELS):
        for idx, plot_i in enumerate(to_plot):
            sns.heatmap(
                train_data[plot_i, :, :, channel_i],
                ax=axarr[channel_i, idx * 2],
                cbar=False,
            )
            sns.heatmap(
                test_data[plot_i, :, :, channel_i],
                ax=axarr[channel_i, (idx * 2) + 1],
                cbar=False,
            )
    # for idx in range(n):
    #     axarr[idx * 2].set_title("R")
    #     axarr[(idx * 2) + 1].set_title("G")
    f.tight_layout()
    f.savefig("training.png", dpi=200)

    if args.run_sweep:
        sweep_id = wandb.sweep(sweep_config, project="sweeps-demo")
        wandb.agent(sweep_id, train_model_wandb, count=args.search_size)
    
    else:
        train_ds, test_ds = build_data()
        model = build_model(fc_layers=0, latent_dimensions=8)
        optimizer = build_optimizer()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        loss_object = tf.keras.losses.MeanSquaredError()

        res = []

        prev_loss = np.inf
        bad_epochs = 0

        for epoch in range(10):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            test_loss.reset_states()

            for images in train_ds:
                train_step(images, images, model, optimizer, loss_object, train_loss)
            for images in test_ds:
                test_step(images, images, model, loss_object, test_loss)

            validation_loss = test_loss.result()

            if validation_loss < prev_loss:
                prev_loss = validation_loss
                bad_epochs = 0
            else:
                # count the number of epochs since we decreased the validation loss
                if bad_epochs > 5:
                    break
                else:
                    bad_epochs += 1
                    prev_loss = validation_loss

            print(
                    f'Epoch {epoch + 1}, '
                    f'Loss: {train_loss.result()}, '
                    f'Test Loss: {validation_loss}, '
            )
            res.append({"Training loss": train_loss.result(),
                        "Validation loss": validation_loss,
                        "Epoch": epoch + 1})

        res_df = pd.DataFrame(res)

        f, ax = plt.subplots()
        ax.plot(res_df["Training loss"], label="Training Loss")
        ax.plot(res_df["Validation loss"], label="Validation Loss")
        ax.legend()
        f.savefig("loss.png", dpi=200)

        # create some new simulated data. we'll use these data to figure out the reconstruction
        # error of the model on unseen data.

        root_dist = np.array([0.25, 0.25, 0.25, 0.25])

        sim = simulation.simulate_exp

        # first, initialize generator object
        gen = generator.Generator(
            sim,
            global_vars.NUM_HAPLOTYPES // 2,
            np.random.randint(1, 2**32),
)

        # simulate a bunch of training examples
        normal_data = gen.simulate_batch(10, root_dist, mutator_threshold=0)
        #mutator_data = gen.simulate_batch(100, root_dist, mutator_threshold=1)

        encoded_data = model.encoder(normal_data).numpy()
        decoded_data = model.decoder(encoded_data).numpy()

        f, ax = plt.subplots()
        n = 2
        to_plot = np.arange(n)
        f, axarr = plt.subplots(global_vars.NUM_CHANNELS, n * 2, figsize=(8, n * 6))
        for channel_i in np.arange(global_vars.NUM_CHANNELS):
            for idx, plot_i in enumerate(to_plot):
                sns.heatmap(normal_data[plot_i, :, :, channel_i], ax=axarr[channel_i, idx * 2], cbar=False)
                sns.heatmap(decoded_data[plot_i, :, :, channel_i], ax=axarr[channel_i, (idx * 2) + 1], cbar=False)
        # for idx in range(n):
        #     axarr[idx * 2].set_title("R")
        #     axarr[(idx * 2) + 1].set_title("G")
        f.tight_layout()
        f.savefig("real_vs_decoded.png", dpi=200)
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-run_sweep", action="store_true")
    p.add_argument("-search_size", type=int, default=20)
    args = p.parse_args()
    main(args)







# # create some new simulated data. we'll use these data to figure out the reconstruction
# # error of the model on unseen data.

# # simulate a bunch of training examples
# normal_data = gen.simulate_batch(100, root_dist, mutator_threshold=0)
# mutator_data = gen.simulate_batch(100, root_dist, mutator_threshold=1)

# encoded_data = model.encoder(normal_data).numpy()
# decoded_data = model.decoder(encoded_data).numpy()

# f, ax = plt.subplots()
# n = 2
# to_plot = np.arange(n)
# f, axarr = plt.subplots(global_vars.NUM_CHANNELS, n * 2, figsize=(8, n * 6))
# for channel_i in np.arange(global_vars.NUM_CHANNELS):
#     for idx, plot_i in enumerate(to_plot):
#         sns.heatmap(normal_data[plot_i, :, :, channel_i], ax=axarr[channel_i, idx * 2], cbar=False)
#         sns.heatmap(decoded_data[plot_i, :, :, channel_i], ax=axarr[channel_i, (idx * 2) + 1], cbar=False)
# # for idx in range(n):
# #     axarr[idx * 2].set_title("R")
# #     axarr[(idx * 2) + 1].set_title("G")
# f.tight_layout()
# f.savefig("real_vs_decoded.png", dpi=200)

# normal_reconstructions = model.predict(normal_data)
# mutator_reconstructions = model.predict(mutator_data)
# normal_loss = tf.reduce_mean(losses.mse(normal_reconstructions, normal_data), axis=(1, 2)).numpy()
# mutator_loss = tf.reduce_mean(losses.mse(mutator_reconstructions, mutator_data), axis=(1, 2)).numpy()

# max_loss = np.max([np.max(normal_loss), np.max(mutator_loss)])
# min_loss = np.min([np.min(normal_loss), np.min(mutator_loss)])

# bins = np.linspace(min_loss, max_loss, num=50)
# f, ax = plt.subplots()
# ax.hist(normal_loss, bins=bins, alpha=0.5, label="Normal")
# ax.hist(mutator_loss, bins=bins, alpha=0.5, label="Mutator")
# ax.legend()
# ax.set_xlabel("Loss")
# ax.set_ylabel("No of examples")
# f.savefig("recon_loss.png", dpi=200)