import generator
import simulation
import global_vars
import autoencoder
import util
import context_encoder

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
from sklearn.metrics import roc_curve
from tensorflow.keras.datasets import fashion_mnist


def build_model(
    input_shape: Tuple[int] = (
        1,
        global_vars.NUM_SNPS,
        global_vars.NUM_CHANNELS,
    ),
    activation: str = "elu",
    latent_dimensions: int = 8,
):
    model = context_encoder.ContextEncoderOneHot(
        input_shape=input_shape,
        activation=activation,
        latent_dimensions=latent_dimensions,
    )

    model.build_graph(
        (
            1,
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
    )
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


def measure_disc_loss(discriminator, y_pred, y_true):
    # calculate adversarial loss. take the average of the loss of the
    # discriminator on both real and predicted fills
    true_disc_prediction = discriminator(y_true, training=True)
    true_disc_labels = tf.ones_like(true_disc_prediction)
    true_disc_loss = tf.keras.losses.binary_crossentropy(
        true_disc_labels,
        true_disc_prediction,
        from_logits=True,
    )

    fake_disc_prediction = discriminator(y_pred, training=True)
    fake_disc_labels = tf.ones_like(fake_disc_prediction)
    fake_disc_loss = tf.keras.losses.binary_crossentropy(
        fake_disc_labels,
        fake_disc_prediction,
        from_logits=True,
    )

    return (true_disc_loss + fake_disc_loss) / 2.0


def measure_recon_loss(y_true, y_pred):
    return tf.math.reduce_mean(
        tf.keras.losses.mean_squared_error(y_true, y_pred), axis=(1, 2)
    )


def measure_total_loss(discriminator, y_true, y_pred, l2_weight: float = 0.999):
    disc_loss = measure_disc_loss(discriminator, y_pred, y_true)
    recon_loss = measure_recon_loss(y_true, y_pred)

    return (recon_loss * l2_weight) + (disc_loss * (1 - l2_weight))


def train_step(
    context_encoder,
    discriminator,
    ce_opt,
    disc_opt,
    X: np.ndarray,
    y: np.ndarray,
    l2_weight: float = 0.999,
):
    with tf.GradientTape(persistent=True) as tape:
        predicted_fill = context_encoder(X, training=True)
        # calculate simple L2 loss between predicted and true fill
        l2_loss = measure_recon_loss(y, predicted_fill)
        disc_loss = measure_disc_loss(discriminator, predicted_fill, y)
        # calculate total loss that we'll use to update the context encoder
        total_loss = (l2_loss * l2_weight) + (disc_loss * (1 - l2_weight))

    # update discriminator using discriminator loss (so that discriminator gets better)
    disc_grads = tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # update CE using CE loss (so that CE gets better)
    ce_grads = tape.gradient(l2_loss, context_encoder.trainable_variables)
    ce_opt.apply_gradients(zip(ce_grads, context_encoder.trainable_variables))

    return disc_loss, l2_loss, total_loss


def main(args):
    with np.load("data/data.npz") as data:
        train_data = data["train"]
        train_labels = data["labels"]

    # split out individual rows (i.e., haplotypes) into new arrays
    train_data = np.concatenate(train_data, axis=2)
    train_data = np.transpose(train_data, (2, 0, 1, 3))

    input_shape = train_data.shape

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        train_data,
        train_labels,
        test_size=0.2,
        shuffle=True,
    )

    quarter_idx = global_vars.NUM_SNPS // 4

    f, axarr = plt.subplots(
        global_vars.NUM_CHANNELS,
        2,
        figsize=(16, 8),
        sharex=True,
        sharey=True,
    )
    # labels = ["REF", "ALT", "Distance"]
    for channel_i in range(global_vars.NUM_CHANNELS):
        sub = X_test[0, :, :, channel_i].copy()
        sns.heatmap(sub, ax=axarr[channel_i, 0], cbar=False)
        sub[:, quarter_idx : quarter_idx * 3] = 0
        sns.heatmap(sub, ax=axarr[channel_i, 1], cbar=False)

    axarr[0, 0].set_title("Original")
    axarr[0, 1].set_title("Masked")

    f.tight_layout()
    f.savefig("orig.png", dpi=200)

    model = build_model(
        input_shape=input_shape[1:],
        activation="relu",
        latent_dimensions=100,
    )

    disc = context_encoder.DiscriminatorOneHot()
    disc.build_graph(
        (
            1,
            4,
            global_vars.NUM_SNPS // 2,
            global_vars.NUM_CHANNELS,
        )
    )
    disc.discriminator.summary()

    # context encoder LR is 10x higher
    ce_optimizer = build_optimizer(learning_rate=1e-3)
    disc_optimizer = build_optimizer(learning_rate=1e-4)

    EPOCHS = 20
    BATCH_SIZE = 64

    res = []

    for epoch in tqdm.tqdm(range(EPOCHS)):
        for step in range(X_train.shape[0] // BATCH_SIZE):
            start_idx = BATCH_SIZE * step
            end_idx = BATCH_SIZE * (step + 1)
            X_batch = X_train[start_idx:end_idx].copy()
            y_batch = X_train[start_idx:end_idx].copy()

            # subset y data to only include the middle chunk of image
            y_batch = y_batch[
                :, :, quarter_idx : quarter_idx * 3, :
            ]

            disc_loss, recon_loss, total_loss = train_step(
                model,
                disc,
                ce_optimizer,
                disc_optimizer,
                X_batch,
                y_batch,
            )

        # check out the model at each step
        predictions = model.predict(X_test)

        # measure test loss
        disc_loss_test = tf.math.reduce_sum(
            measure_disc_loss(
                disc,
                predictions,
                X_test[
                    :, :, quarter_idx : quarter_idx * 3, :
                ],
            )
        ).numpy()
        recon_loss_test = tf.math.reduce_sum(
            measure_recon_loss(
                predictions,
                X_test[
                    :, :, quarter_idx : quarter_idx * 3, :
                ],
            )
        ).numpy()

        for loss, label in zip(
            (disc_loss, recon_loss, total_loss),
            ("disc", "recon", "total"),
        ):
            res.append(
                {
                    "epoch": epoch,
                    "loss": tf.math.reduce_sum(loss).numpy(),
                    "loss_kind": label,
                    "group": "training",
                }
            )
        res.append(
            {
                "epoch": epoch,
                "loss": disc_loss_test,
                "loss_kind": "disc",
                "group": "validation",
            }
        )
        res.append(
            {
                "epoch": epoch,
                "loss": recon_loss_test,
                "loss_kind": "recon",
                "group": "validation",
            }
        )

        f, axarr = plt.subplots(
            global_vars.NUM_CHANNELS, 2, figsize=(16, 8), sharey=True
        )
        for channel_i in range(global_vars.NUM_CHANNELS):
            sub = X_test[0, :, :, channel_i].copy()
            sns.heatmap(sub, ax=axarr[channel_i, 0], cbar=False)
            sub[:, quarter_idx : quarter_idx * 3] = 0
            sub[
                :, quarter_idx : quarter_idx * 3
            ] += predictions[0, :, :, channel_i]
            sns.heatmap(sub, ax=axarr[channel_i, 1], cbar=False)
            # for i in (0, 1):
            # axarr[channel_i, i].set_yticklabels(["A", "T", "C", "G"])
            # axarr[channel_i, i].set_ylabel(labels[channel_i], rotation=0)
            # axarr[channel_i, i].set_xlabel("SNP index")

        axarr[0, 0].set_title("Original")
        axarr[0, 1].set_title("Masked")

        f.tight_layout()
        f.savefig("preds.png", dpi=200)

    # loss_dist = measure_total_loss(disc, X_test[:, :, quarter_idx : quarter_idx * 3, :], predictions)
    loss_dist = measure_recon_loss(
        X_test[:, :, quarter_idx : quarter_idx * 3, :],
        predictions,
    )
    loss = []

    fpr, tpr, thresholds = roc_curve(y_test, loss_dist.numpy())
    f, ax = plt.subplots()
    ax.plot(fpr, tpr)
    f.savefig("roc.png")

    f, ax = plt.subplots()

    for label in np.unique(y_test):
        label_idxs = np.where(y_test == label)[0]
        for i in label_idxs:
            loss.append({"label": label, "loss": loss_dist.numpy()[i]})
    loss = pd.DataFrame(loss)
    sns.boxplot(data=loss, x="label", y="loss", ax=ax, color="white")
    sns.stripplot(data=loss, x="label", y="loss", ax=ax)
    f.savefig("dist.png", dpi=200)

    res = pd.DataFrame(res)

    g = sns.FacetGrid(data=res, col="group", sharey=False)
    g.map(sns.lineplot, "epoch", "loss", "loss_kind")
    f.tight_layout()
    g.savefig("loss.png", dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-run_sweep", action="store_true")
    p.add_argument("-search_size", type=int, default=20)
    args = p.parse_args()
    main(args)
