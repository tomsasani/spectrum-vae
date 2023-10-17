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

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

### BUILD MODEL
def build_model(
    # initial_filters: int = 8,
    # conv_layers: int = 2,
    # conv_layer_multiplier: int = 2,
    latent_dimensions: int = 8,
    fc_layers: int = 0,
    fc_layer_size: int = 32,
    dropout: float = 0.5,
    apply_nin: bool = False,
    nin_filters: int = 32,
    activation: str = "elu",
    #kernel_size: int = 5,
):


    model = autoencoder.AESimple(
        # initial_filters=initial_filters,
        # conv_layers=conv_layers,
        # conv_layer_multiplier=conv_layer_multiplier,
        latent_dimensions=latent_dimensions,
        fc_layers=fc_layers,
        fc_layer_size=fc_layer_size,
        dropout=dropout,
        apply_nin=apply_nin,
        nin_filters=nin_filters,
        activation=activation,
        #kernel_size=kernel_size,
    )
    model.build_graph((
        1,
        1,
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


def build_data(batch_size: int = 32, sort: bool = True):
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
            # latent_dimensions=config.latent_dimensions,
            #fc_layers=config.fc_layers,
            #fc_layer_size=config.fc_layer_size,
            #dropout=config.dropout,
            apply_nin=config.apply_nin,
            nin_filters=config.nin_filters,
            activation=config.activation,
            #kernel_size=config.kernel_size,
        )
        optimizer = build_optimizer(
            learning_rate=config.learning_rate,
            #decay_rate=config.decay_rate,
            #decay_steps=config.decay_steps,
        )

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

            wandb.log({"validation_loss": validation_loss})

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
        'apply_nin': {
            'values': [True, False]
        },
        'nin_filters': {
            'values': [16, 32]
        },
        'initial_filters': {
            'values': [8, 16, 32]
        },
        'conv_layers': {
            'values': [3]
        },
        'conv_layer_multiplier': {
            'values': [1, 2]
        },
        # 'latent_dimensions': {
        #     'values': [4, 8, 16, 32, 64]
        # },
        # 'fc_layers': {
        #     'values': [1, 2, 3, 4]
        # },
        # 'fc_layer_size': {
        #     'values': [16, 32, 64, 128]
        # },
        # 'dropout': {
        #     'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        # },
        'activation': {
            'values': ['elu', 'relu']
        },
        'learning_rate': {
            'values': [1e-4, 5e-4, 1e-3, 5e-3]
        },
        # 'decay_rate': {
        #     'values': [0.92, 0.94, 0.96, 0.98]
        # },
        # 'decay_steps': {
        #     'values': [10]
        # },
        'batch_size': {
            'values': [8, 16, 32],
        },
        # 'kernel_size': {
        #     'values': [5, 7]
        # },
        'epochs': {
            'value': 10
        }
    }

    sweep_config['parameters'] = parameters_dict


    if args.run_sweep:
        sweep_id = wandb.sweep(sweep_config, project="sweeps-demo")
        wandb.agent(sweep_id, train_model_wandb, count=args.search_size)

    else:

        with np.load("data/data.npz") as data:
            train_data = data["train"]
            test_data = data["test"]


        f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 1, figsize=(12, global_vars.NUM_CHANNELS * 2))
        hap = train_data[0, :, :, :]
        for channel_i in np.arange(global_vars.NUM_CHANNELS):
            sns.heatmap(hap[:, :, channel_i], ax=axarr[channel_i], cbar=True)
        f.tight_layout()
        f.savefig("training.png", dpi=200)

        train_ds, test_ds = build_data(
            batch_size=8,
            sort=False,
        )
        model = build_model(
            # conv_layers=3,
            # conv_layer_multiplier=1,
            # initial_filters=8,
            fc_layers=3,
            latent_dimensions=8,
            dropout=0.5,
            fc_layer_size=128,
            apply_nin=False,
            nin_filters=16,
            activation='elu',
            #kernel_size=5,
        )
        optimizer = build_optimizer(learning_rate=5e-3)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        loss_object = tf.keras.losses.MeanSquaredError()

        res = []

        prev_loss = np.inf
        bad_epochs = 0
        total_epochs = 0
        while bad_epochs < 5 and total_epochs < 20:

            #for epoch in range(EPOCHS):
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

        root_dist = np.array([0.25, 0.25, 0.25, 0.25])

        sim = simulation.simulate_exp

        # first, initialize generator object
        gen = generator.Generator(
            sim,
            [global_vars.NUM_HAPLOTYPES],
            np.random.randint(1, 2**32),
        )

        N_ROC = 1_000

        # simulate a bunch of training examples
        normal_data = gen.simulate_batch(N_ROC, root_dist, mutator_threshold=0)
        mutator_data = gen.simulate_batch(N_ROC, root_dist, mutator_threshold=1)

        #normal_data_split = np.concatenate(np.split(normal_data, global_vars.NUM_HAPLOTYPES, axis=1))
        #mutator_data_split = np.concatenate(np.split(mutator_data, global_vars.NUM_HAPLOTYPES, axis=1))
        # to_predict = np.expand_dims(normal_data[:, 0, :, :], axis=0)
        encoded_data = model.encoder(normal_data).numpy()
        decoded_data = model.decoder(encoded_data).numpy()

        f, axarr = plt.subplots(global_vars.NUM_CHANNELS, 2, figsize=(16, global_vars.NUM_CHANNELS * 2))
        for channel_i in np.arange(global_vars.NUM_CHANNELS):
            sns.heatmap(normal_data[0, :, :, channel_i], ax=axarr[channel_i, 0], cbar=True)
            sns.heatmap(decoded_data[0, :, :, channel_i], ax=axarr[channel_i, 1], cbar=True)

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