import generator
import simulation
import global_vars
import autoencoder
import util

from typing import Tuple

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
from tensorflow.keras.datasets import fashion_mnist


def sort_batch(batch: np.ndarray):
    sorted_batch = np.zeros(batch.shape)
    for batch_i in range(batch.shape[0]):
        arr = batch[batch_i, :, :, :]
        # sort the array by genetic similarity
        sorted_idxs = util.sort_min_diff(arr)
        sorted_batch[batch_i] = arr[sorted_idxs, :, :]
    return sorted_batch

### BUILD MODEL
def build_model(
    input_shape: Tuple[int] = (
        global_vars.NUM_HAPLOTYPES,
        global_vars.NUM_SNPS,
        global_vars.NUM_CHANNELS,
    ),
    initial_filters: int = 8,
    conv_layers: int = 2,
    conv_layer_multiplier: int = 2,
    activation: str = "elu",
    kernel_size: int = 5,
    conv_operations: int = 2,
    latent_dimensions: int = 8,
    fc_layers: int = 2,
    fc_layer_size: int = 64,
    dropout: float = 0.5,
    batch_norm: bool = False,
):

    model = autoencoder.CAEPadded(
        input_shape=input_shape,
        initial_filters=initial_filters,
        conv_layers=conv_layers,
        conv_layer_multiplier=conv_layer_multiplier,
        activation=activation,
        kernel_size=kernel_size,
        conv_operations=conv_operations,
        fc_layer_size=fc_layer_size,
        fc_layers=fc_layers,
        latent_dimensions=latent_dimensions,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    model.build_graph((
        1,
        input_shape[0],
        input_shape[1],
        input_shape[2],
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


def train_model_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config

        with np.load("data/data.npz") as data:
            train = data["train"]

        shape = train.shape

        if config.sort_data:
            train = sort_batch(train)

        # pad if necessary
        padding = 0
        cur_height = shape[1]
        while cur_height % (2 ** config.conv_layers) != 0:
            padding += 1
            cur_height += 1
        padding_zeros = np.zeros((shape[0], padding, shape[2], shape[3]))
        train_data = np.concatenate((train, padding_zeros), axis=1)

        X_train, X_test = sklearn.model_selection.train_test_split(
            train_data,
            test_size=0.2,
        )

        input_shape = X_train.shape

        model = build_model(
            input_shape=input_shape[1:],
            conv_layers=config.conv_layers,
            conv_layer_multiplier=config.conv_layer_multiplier,
            initial_filters=config.initial_filters,
            activation=config.activation,
            kernel_size=config.kernel_size,
            conv_operations=config.conv_operations,
            latent_dimensions=0,
            fc_layers=0,
            fc_layer_size=0,
            dropout=0,
            batch_norm=config.batch_norm,
        )
        optimizer = build_optimizer(
            learning_rate=config.learning_rate,
            decay_rate=config.decay_rate,
            decay_steps=config.decay_steps,
        )

        model.compile(optimizer=optimizer, loss=losses.MeanSquaredError())
        history = model.fit(X_train, X_train,
                epochs=config.epochs,
                shuffle=True,
                validation_data=(X_test, X_test),
                batch_size=config.batch_size,
                )

        validation_loss = history.history["loss"]


        for i in range(len(validation_loss)):
            wandb.log({"validation_loss": validation_loss[i]})

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
        'latent_dimensions': {
            'value': 0
        },
        'initial_filters': {
            'values': [8, 16]
        },
        'conv_layers': {
            'values': [3, 4],
        },
        'conv_layer_multiplier': {
            'values': [1, 2]
        },
        'conv_operations': {
            'values': [1, 2, 3],
        },
        'fc_layers': {
            'value': 0
        },
        'fc_layer_size': {
            'value': 128
        },
        'dropout': {
            'value': 0.1
        },
        'activation': {
            'values': ['relu', 'elu']
        },
        'learning_rate': {
            'value': 1e-4, #[1e-3, 5e-3, 1e-4, 5e-4]
        },
        'decay_rate': {
            'value': 0.92, #[0.92, 0.94, 0.96, 0.98],
        },
        'decay_steps': {
            'value': 10
        },
        'batch_size': {
            'values': [16, 32]
        },
        'kernel_size': {
            'values': [3, 5]
        },
        'epochs': {
            'value': 10
        },
        'sort_data': {
            'value': True,
        },
        'batch_norm': {
            'values': [True, False]
        }
    }

    sweep_config['parameters'] = parameters_dict

    BATCH_SIZE = 32

    if args.run_sweep:
        sweep_id = wandb.sweep(sweep_config, project="sweeps-demo")
        wandb.agent(sweep_id, train_model_wandb, count=args.search_size)

    else:

        with np.load("data/data.npz") as data:
            train_data = data["train"]

        CONV_LAYERS = 3


        train_data = sort_batch(train_data)
        shape = train_data.shape

        # pad if necessary
        padding = 0
        cur_height = shape[1]
        while cur_height % (2 ** CONV_LAYERS) != 0:
            padding += 1
            cur_height += 1
        padding_zeros = np.zeros((shape[0], padding, shape[2], shape[3]))
        train_data = np.concatenate((train_data, padding_zeros), axis=1)

        input_shape = train_data.shape

        X_train, X_test = sklearn.model_selection.train_test_split(
            train_data,
            test_size=0.2,
        )

        # (train_data, _), (test_data, _) = fashion_mnist.load_data()

        # train_data = train_data.astype('float32') / 255.
        # test_data = test_data.astype('float32') / 255.

        # train_data = train_data[:1000, :, :]
        # test_data = test_data[:250, :, :]

        idx = np.random.randint(X_train.shape[0])
        f, axarr = plt.subplots(global_vars.NUM_CHANNELS, figsize=(12, 8))
        haps = X_train[idx, :, :, :]
        for channel_i in range(global_vars.NUM_CHANNELS):
            sns.heatmap(haps[:, :, channel_i], ax=axarr[channel_i], cbar=True)
        f.tight_layout()
        f.savefig("training.png", dpi=200)

        model = build_model(
            input_shape=input_shape[1:],
            conv_layers=CONV_LAYERS,
            conv_layer_multiplier=1,
            initial_filters=32,
            activation='elu',
            kernel_size=3,
            conv_operations=1,
            fc_layers=0,
            fc_layer_size=256,
            dropout=0.2,
            latent_dimensions=0,
        )

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        optimizer = build_optimizer(learning_rate=5e-4)

        model.compile(optimizer=optimizer, loss=losses.MeanSquaredError())

        history = model.fit(X_train, X_train,
                epochs=5,
                shuffle=True,
                validation_data = (X_test, X_test),
                callbacks=[callback],
                batch_size=BATCH_SIZE,
                )

        f, ax = plt.subplots()
        ax.plot(history.history["loss"], label="loss")
        ax.plot(history.history["val_loss"], label="val_loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel("Loss")
        ax.legend()

        f.savefig("acc.png", dpi=200)

        decoded_imgs = model.predict(X_test)#, batch_size=BATCH_SIZE)

        idx = np.random.randint(X_test.shape[0])
        f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 2, figsize=(12, 12))
        for channel_i in range(global_vars.NUM_CHANNELS):
            sns.heatmap(X_test[idx, :, :, channel_i], ax=axarr[channel_i, 0])
            sns.heatmap(decoded_imgs[idx, :, :, channel_i], ax=axarr[channel_i, 1])
        f.tight_layout()
        f.savefig("encoded.png")



        # create some new simulated data. we'll use these data to figure out the reconstruction
        # error of the model on unseen data.

        # N_ROC = 200

        # root_dist = np.array([0.25, 0.25, 0.25, 0.25])

        # sim = simulation.simulate_exp_one_channel

        # # first, initialize generator object
        # gen = generator.Generator(
        #     sim,
        #     [global_vars.NUM_HAPLOTYPES // 2],
        #     np.random.randint(1, 2**32),
        # )

        # # simulate a bunch of training examples
        # normal_data = gen.simulate_batch(
        #     N_ROC,
        #     root_dist,
        #     mutator_threshold=0,
        # )
        # mutator_data = gen.simulate_batch(
        #     N_ROC,
        #     root_dist,
        #     mutator_threshold=1,
        # )

        # A = 10
        # B = 1_000 - A

        # print (normal_data.shape)

        # mixture_a = gen.simulate_batch(A, root_dist, mutator_threshold=1)
        # mixture_b = gen.simulate_batch(B, root_dist, mutator_threshold=0)
        # mixture_data = np.concatenate((mixture_a, mixture_b), axis=0)

        with np.load("data/labeled.npz") as data:
            normal_data = data["neg"]
            mutator_data = data["pos"]

        normal_data = np.concatenate((normal_data, padding_zeros), axis=1)
        mutator_data = np.concatenate((mutator_data, padding_zeros), axis=1)

        normal_reconstructions = model.predict(normal_data, batch_size=BATCH_SIZE)
        mutator_reconstructions = model.predict(mutator_data, batch_size=BATCH_SIZE)

        #mixture_reconstructions = model.predict(mixture_data, batch_size=BATCH_SIZE)

        normal_loss = tf.reduce_mean(losses.mse(normal_reconstructions, normal_data), axis=(1, 2)).numpy()
        mutator_loss = tf.reduce_mean(losses.mse(mutator_reconstructions, mutator_data), axis=(1, 2)).numpy()

        # mixture_loss = tf.reduce_mean(losses.mse(mixture_reconstructions, mixture_data), axis=(1, 2)).numpy()

        # res = []
        # labels = [1] * A
        # labels += [0] * B
        # for i in range(mixture_loss.shape[0]):
        #     label = labels[i]
        #     loss = mixture_loss[i]
        #     res.append({"loss": loss, "label": label})
        # res_df = pd.DataFrame(res)

        # f, ax = plt.subplots()
        # sns.stripplot(data=res_df, x="label", y="loss", ax=ax)
        # f.savefig("mixture.png")

        max_loss = np.max([np.max(normal_loss), np.max(mutator_loss)])
        min_loss = np.min([np.min(normal_loss), np.min(mutator_loss)])

        bins = np.linspace(min_loss, max_loss, num=50)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.hist(normal_loss, bins=bins, alpha=0.25, label="Normal")
        ax1.hist(mutator_loss, bins=bins, alpha=0.25, label="Mutator")
        ax1.legend()
        ax1.set_xlabel("Loss")
        ax1.set_ylabel("No of examples")

        y_true = [1] * normal_data.shape[0]
        y_true += [0] * mutator_data.shape[0]
        y_pred = np.concatenate((normal_loss, mutator_loss))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(np.array(y_true), y_pred)
        ax2.plot(fpr, tpr)

        ax2.plot([0, 1], [0, 1], c="k", ls=":")

        auc = sklearn.metrics.auc(fpr, tpr)
        ax2.set_title(f"AUC = {round(auc, 3)}")
        f.tight_layout()
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