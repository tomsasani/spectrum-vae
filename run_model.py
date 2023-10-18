import generator
import simulation
import global_vars
import autoencoder
import util

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import losses
import tensorflow as tf
import pandas as pd
import wandb
import argparse
import sklearn
import tqdm


### BUILD MODEL
def build_model(
    initial_filters: int = 8,
    conv_layers: int = 2,
    conv_layer_multiplier: int = 2,
    activation: str = "elu",
    kernel_size: int = 5,
    fc_layers: int = 2,
    fc_layer_size: int = 64,
    dropout: float = 0.5,
    latent_dimensions: int = 2,
):

    model = autoencoder.CAESimple(
        initial_filters=initial_filters,
        conv_layers=conv_layers,
        conv_layer_multiplier=conv_layer_multiplier,
        activation=activation,
        kernel_size=kernel_size,
        fc_layer_size=fc_layer_size,
        fc_layers=fc_layers,
        dropout=dropout,
        latent_dimensions=latent_dimensions,
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


def build_optimizer(
    learning_rate: float = 1e-4,
    decay_rate: float = 0.96,
    decay_steps: int = 10,
):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
    return optimizer

def sort_batch(batch: np.ndarray):
    sorted_batch = np.zeros(batch.shape)
    for batch_i in range(batch.shape[0]):
        arr = batch[batch_i, :, :, :]
        # sort the array by genetic similarity
        sorted_idxs = util.sort_min_diff(arr)
        sorted_batch[batch_i] = arr[sorted_idxs, :, :]
    return sorted_batch


def build_data(batch_size: int = 32, sort: bool = False):
    train_data, test_data = None, None
    with np.load("data/data.npz") as data:
        train_data = data["train"]
        test_data = data["test"]
        if sort:
            print ("SORTING")
            train_data = sort_batch(train_data)
            test_data = sort_batch(test_data)
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

        train_ds, test_ds = build_data(
            batch_size=config.batch_size,
        )
        model = build_model(
            conv_layers=config.conv_layers,
            conv_layer_multiplier=config.conv_layer_multiplier,
            initial_filters=config.initial_filters,
            latent_dimensions=config.latent_dimensions,
            fc_layers=config.fc_layers,
            fc_layer_size=config.fc_layer_size,
            dropout=config.dropout,
            activation=config.activation,
            kernel_size=config.kernel_size,
        )
        optimizer = build_optimizer(
            learning_rate=config.learning_rate,
            decay_rate=config.decay_rate,
            decay_steps=config.decay_steps,
        )

        loss_object = tf.keras.losses.MeanSquaredError()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        for epoch in tqdm.tqdm(range(config.epochs)):
            train_loss.reset_states()
            test_loss.reset_states()

            for images in train_ds:
                train_step(images, images, model, optimizer, loss_object, train_loss)
            for images in test_ds:
                test_step(images, images, model, loss_object, test_loss)

            validation_loss = test_loss.result()

            wandb.log({"validation_loss": validation_loss})

def main(args):

    ### define the sweep
    sweep_config = {"method": "random"}

    metric = {'name': 'validation_loss', 'goal': 'minimize'}

    sweep_config['metric'] = metric

    # set up parameters over which to sweep
    parameters_dict = {
        'optimizer': {
            'value': 'adam'
        },
        
        'initial_filters': {
            'values': [8, 16]
        },
        'conv_layers': {
            'value': 3
        },
        'conv_layer_multiplier': {
            'value': 1
        },
        'latent_dimensions': {
            'values': [2, 8, 32, 128],
        },
        'fc_layers': {
            'values': [2, 3, 4]
        },
        'fc_layer_size': {
            'values': [16, 32, 64, 128]
        },
        'dropout': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'activation': {
            'values': ['elu', 'relu']
        },
        'learning_rate': {
            'values': [1e-4, 5e-4, 1e-3, 5e-3]
        },
        'decay_rate': {
            'values': [0.92, 0.94, 0.96, 0.98]
        },
        'decay_steps': {
            'value': 10
        },
        'batch_size': {
            'values': [8, 16, 32],
        },
        'kernel_size': {
            'values': [5, 7]
        },
        'epochs': {
            'value': 20
        }
    }

    sweep_config['parameters'] = parameters_dict


    if args.run_sweep:
        sweep_id = wandb.sweep(sweep_config, project="sweeps-demo")
        wandb.agent(sweep_id, train_model_wandb, count=args.search_size)

    else:

        SORT = False

        with np.load("data/data.npz") as data:
            train_data = data["train"]
            test_data = data["test"]

        if SORT:
            train_data = sort_batch(train_data)
            test_data = sort_batch(test_data)

        f, ax = plt.subplots(figsize=(12, 6))
        hap = train_data[0, :, :, :]
        sns.heatmap(hap[:, :, 0], ax=ax, cbar=True)
        f.tight_layout()
        f.savefig("training.png", dpi=200)

        train_ds, test_ds = build_data(
            batch_size=16,
            sort=SORT,
        )
        model = build_model(
            conv_layers=3,
            conv_layer_multiplier=1,
            initial_filters=8,
            activation='elu',
            kernel_size=5,
            fc_layers=2,
            fc_layer_size=64,
            latent_dimensions=32,
            dropout=0.5,
        )
        optimizer = build_optimizer(learning_rate=5e-3, decay_rate=0.92, decay_steps=10)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        loss_object = tf.keras.losses.MeanSquaredError()

        root_dist = np.array([0.25, 0.25, 0.25, 0.25])

        sim = simulation.simulate_exp

        # first, initialize generator object
        gen = generator.Generator(
            sim,
            [global_vars.NUM_HAPLOTYPES // 2],
            np.random.randint(1, 2**32),
        )

        res = []

        prev_loss = np.inf
        bad_epochs = 0
        total_epochs = 0
        while bad_epochs < 5 and total_epochs < 10:

            train_loss.reset_states()
            test_loss.reset_states()

            # train_data = gen.simulate_batch(1_000, root_dist, mutator_threshold=0)
            # test_data = gen.simulate_batch(1_000, root_dist, mutator_threshold=0)

            # train_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(16)
            # test_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(16)

            for images in tqdm.tqdm(train_ds):
                train_step(images, images, model, optimizer, loss_object, train_loss)
            for images in tqdm.tqdm(test_ds):
                test_step(images, images, model, loss_object, test_loss)

            validation_loss = test_loss.result()

            if validation_loss < prev_loss:
                prev_loss = validation_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                prev_loss = validation_loss

            print(
                    f'Epoch {total_epochs + 1}, '
                    f'Loss: {train_loss.result()}, '
                    f'Test Loss: {validation_loss}, '
            )
            res.append({"Training loss": train_loss.result(),
                        "Validation loss": validation_loss,
                        "Epoch": total_epochs + 1})
            total_epochs += 1

        res_df = pd.DataFrame(res)

        f, ax = plt.subplots()
        ax.plot(res_df["Training loss"], label="Training Loss")
        ax.plot(res_df["Validation loss"], label="Validation Loss")
        ax.legend()
        f.savefig("loss.png", dpi=200)

        # create some new simulated data. we'll use these data to figure out the reconstruction
        # error of the model on unseen data.

        

        N_ROC = 200

        # simulate a bunch of training examples
        normal_data = gen.simulate_batch(N_ROC, root_dist, mutator_threshold=0)
        mutator_data = gen.simulate_batch(N_ROC, root_dist, mutator_threshold=1)
        # sort
        if SORT:
            normal_data_sorted = sort_batch(normal_data)
            mutator_data_sorted = sort_batch(mutator_data)
        else:
            normal_data_sorted = np.copy(normal_data)
            mutator_data_sorted = np.copy(mutator_data)


        #normal_data_split = np.concatenate(np.split(normal_data, global_vars.NUM_HAPLOTYPES, axis=1))
        #mutator_data_split = np.concatenate(np.split(mutator_data, global_vars.NUM_HAPLOTYPES, axis=1))
        # to_predict = np.expand_dims(normal_data[:, 0, :, :], axis=0)
        encoded_data = model.encoder(normal_data_sorted).numpy()
        decoded_data = model.decoder(encoded_data).numpy()

        f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 3, figsize=(16, global_vars.NUM_CHANNELS * 2))
        #for channel_i in range(global_vars.NUM_CHANNELS):
        sns.heatmap(normal_data_sorted[0, :, :, 0], ax=axarr[0], cbar=True)
        sns.heatmap(normal_data_sorted[0, :, :, 0], ax=axarr[1], cbar=True)
        sns.heatmap(decoded_data[0, :, :, 0], ax=axarr[2], cbar=True)

        f.tight_layout()
        f.savefig("real_vs_decoded.png", dpi=200)

        normal_reconstructions = model.predict(normal_data)
        mutator_reconstructions = model.predict(mutator_data)
        normal_loss = tf.reduce_mean(losses.mse(normal_reconstructions, normal_data), axis=(1, 2)).numpy()
        mutator_loss = tf.reduce_mean(losses.mse(mutator_reconstructions, mutator_data), axis=(1, 2)).numpy()

        max_loss = np.max([np.max(normal_loss), np.max(mutator_loss)])
        min_loss = np.min([np.min(normal_loss), np.min(mutator_loss)])

        bins = np.linspace(min_loss, max_loss, num=50)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.hist(normal_loss, bins=bins, alpha=0.25, label="Normal")
        ax1.hist(mutator_loss, bins=bins, alpha=0.25, label="Mutator")
        ax1.legend()
        ax1.set_xlabel("Loss")
        ax1.set_ylabel("No of examples")

        y_true = [1] * N_ROC
        y_true += [0] * N_ROC
        y_pred = np.concatenate((normal_loss, mutator_loss))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_true), y_pred)
        ax2.plot(fpr, tpr)

        # lims = [
        #     np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
        #     np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
        # ]
        ax2.plot([0, 1], [0, 1], c="k", ls=":")

        f.savefig("recon_loss.png", dpi=200)

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