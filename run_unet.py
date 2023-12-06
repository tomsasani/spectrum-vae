import generator
import simulation
import global_vars
import autoencoder
import util
import context_encoder
import unet

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
        global_vars.NUM_HAPLOTYPES,
        global_vars.NUM_SNPS,
        global_vars.INPUT_CHANNELS,
    ),
):
    model = unet.Generator(input_shape=input_shape)

    # model.build_graph(
    #     (
    #         1,
    #         input_shape[0],
    #         input_shape[1],
    #         input_shape[2],
    #     )
    # )
    model.summary()

    return model


def build_optimizer(
    learning_rate: float = 1e-4,
    beta_1: float = 0.5,
    beta_2: float = 0.999,
    decay_rate: float = 0.96,
    decay_steps: int = 10,
):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=lr_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
    )
    return optimizer


def measure_disc_loss(discriminator, y_true, y_pred, reduction: str = "auto"):
    """discriminator loss accepts 2 inputs: real images and generated images.
    the 'true' loss measures sigmoid cross entropy between the real images and
    an array of ones. the 'fake' loss measure cross entropy between the
    generated images and an array of zeros.

    Args:
        discriminator (_type_): _description_
        y_true (_type_): _description_
        y_pred (_type_): _description_
        reduction (str, optional): _description_. Defaults to "auto".

    Returns:
        _type_: _description_
    """
    loss_func = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction=reduction,
    )

    # calculate adversarial loss. take the average of the loss of the
    # discriminator on both real and predicted fills
    true_disc_prediction = discriminator(y_true, training=True)
    true_disc_labels = tf.ones_like(true_disc_prediction)
    true_disc_loss = loss_func(
        true_disc_labels,
        true_disc_prediction,
    )

    fake_disc_prediction = discriminator(y_pred, training=True)
    fake_disc_labels = tf.zeros_like(fake_disc_prediction)
    fake_disc_loss = loss_func(
        fake_disc_labels,
        fake_disc_prediction,
    )

    return true_disc_loss + fake_disc_loss


def measure_generator_loss(
    y_true,
    y_pred,
    disc_y_pred,
    lambda_weight: int = 100,
    reduction: str = "auto",
):
    """generator loss measures two things: sigmoid cross-entropy between the
    discriminator predictions on the generated images and an array of ones,
    as well as the L1 (MAE) error between the generated image and real image.

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        reduction (str, optional): _description_. Defaults to "auto".

    Returns:
        _type_: _description_
    """
    mae_loss_func = tf.keras.losses.MeanAbsoluteError(reduction=reduction)
    bce_loss_func = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction=reduction,
    )

    # measure GAN loss
    gan_loss = bce_loss_func(tf.ones_like(disc_y_pred), disc_y_pred)

    # measure L1 loss
    l1_loss = mae_loss_func(y_true, y_pred)

    total_gen_loss = gan_loss + (lambda_weight * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def train_step(
    generator,
    discriminator,
    ce_opt,
    disc_opt,
    X: np.ndarray,
    y: np.ndarray,
):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # make predictions on the input data
        predicted = generator(X, training=True)

        # apply the discriminator to the generated data
        disc_y_pred = discriminator(predicted, training=True)

        # measure the total loss for the generator
        gen_total_loss, gan_loss, l1_loss = measure_generator_loss(
            y,
            predicted,
            disc_y_pred,
        )
        # measure the total loss for the discriminator
        disc_total_loss = measure_disc_loss(discriminator, y, predicted)

    # update discriminator using discriminator loss (so
    # that discriminator gets better)
    disc_grads = disc_tape.gradient(
        disc_total_loss,
        discriminator.trainable_variables,
    )
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # update CE using CE loss (so that CE gets better)
    ce_grads = gen_tape.gradient(
        gen_total_loss,
        generator.trainable_variables,
    )
    ce_opt.apply_gradients(zip(ce_grads, generator.trainable_variables))

    return disc_total_loss, gen_total_loss, gan_loss, l1_loss


def format_cifar(data: np.ndarray):
    n_examples, _, _, n_channels = data.shape
    new_data = np.zeros((n_examples, 32, 32, n_channels))
    for i in np.arange(n_examples):
        img = data[i]
        norm_img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        norm_img = (2 * norm_img) - 1
        new_data[i] = norm_img
    return new_data


def main():
    CIFAR = True

    if CIFAR:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

        N = 1_000

        # training data are paired grayscale/RGB images

        X_train, y_train = (
            format_cifar(tf.image.rgb_to_grayscale(X_train[:N])),
            format_cifar(X_train[:N]),
        )
        X_test, y_test = (
            format_cifar(tf.image.rgb_to_grayscale(X_test[:N])),
            format_cifar(X_test[:N]),
        )
    else:
        with np.load("data/data.npz") as data:
            train_data = data["train_data"]
            train_labels = data["train_labels"]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            train_data,
            train_labels,
            test_size=0.2,
            shuffle=True,
        )

    input_shape = X_train.shape

    if CIFAR:
        f, axarr = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
        idx = np.random.randint(N)
        axarr[0].imshow(X_train[idx], cmap="gray")
        axarr[1].imshow(y_train[idx])
        f.tight_layout()
        f.savefig("orig.png", dpi=200)
    else:
        f, axarr = plt.subplots(
            2, global_vars.NUM_CHANNELS, figsize=(16, 8), sharey=False
        )

        for channel_i in range(global_vars.NUM_CHANNELS):
            sns.heatmap(
                X_train[0, :, :, channel_i],
                ax=axarr[0, channel_i],
                cbar=False,
            )
            sns.heatmap(
                X_train[0, :, :, channel_i] * (1 - X_train_mask[0, :, :, channel_i]),
                ax=axarr[1, channel_i],
                cbar=False,
            )

        f.tight_layout()
        f.savefig("orig.png", dpi=200)

    model = unet.Generator()
    model.summary()

    disc = unet.Discriminator()
    disc.build_graph(
        (
            1,
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.OUTPUT_CHANNELS,
        )
    )
    disc.discriminator.summary()

    # context encoder LR is 10x higher
    LR = 2e-4
    ce_optimizer = build_optimizer(learning_rate=LR)
    disc_optimizer = build_optimizer(learning_rate=LR)

    EPOCHS = 50
    BATCH_SIZE = 64

    res = []

    for epoch in tqdm.tqdm(range(EPOCHS)):
        for step in range(X_train.shape[0] // BATCH_SIZE):
            start_idx = BATCH_SIZE * step
            end_idx = BATCH_SIZE * (step + 1)

            disc_total_loss, gen_total_loss, gan_loss, l1_loss = train_step(
                model,
                disc,
                ce_optimizer,
                disc_optimizer,
                X_train[start_idx:end_idx],
                y_train[start_idx:end_idx],
            )
        if epoch % 1 == 0:
            print("DISC LOSS", disc_total_loss.numpy())
            print(gen_total_loss.numpy())
            print("GAN LOSS", gan_loss.numpy())
            print(l1_loss.numpy())

        # check out the model at each step
        predictions = model.predict(X_test)

        # make discriminator predictions on the predicted data
        disc_predictions = disc(predictions)

        # measure test loss
        disc_loss_test = measure_disc_loss(disc, y_test, predictions)
        gen_loss_test, gan_loss_test, l1_loss_test = measure_generator_loss(
            y_test,
            predictions,
            disc_predictions,
        )

        for loss, label, group in zip(
            (
                disc_total_loss,
                gen_total_loss,
                disc_loss_test,
                gen_loss_test,
            ),
            ("disc", "gen", "disc", "gen"),
            (
                "training",
                "training",
                "validation",
                "validation",
            ),
        ):
            res.append(
                {
                    "epoch": epoch,
                    "loss": loss.numpy(),
                    "loss_kind": label,
                    "group": group,
                }
            )

        if epoch % 5 == 0:
            if CIFAR:
                f, axarr = plt.subplots(3, 4, figsize=(28, 20), sharey=True)
                for plot_idx, idx in enumerate(
                    np.random.randint(X_test.shape[0], size=4)
                ):
                    axarr[0, plot_idx].imshow(X_test[idx], cmap="gray")
                    axarr[1, plot_idx].imshow(predictions[idx])
                    axarr[2, plot_idx].imshow(y_test[idx])

                f.tight_layout()
                f.savefig(f"img/preds.{epoch}.png", dpi=200)
            else:
                f, axarr = plt.subplots(
                    global_vars.NUM_CHANNELS, 3, figsize=(28, 15), sharey=True
                )
                I = np.random.randint(X_test.shape[0])
                for channel_i in range(global_vars.NUM_CHANNELS):
                    # plot original image
                    sns.heatmap(
                        X_test[I, :, :, channel_i],
                        ax=axarr[channel_i, 0],
                        cbar=False,
                        vmin=-1,
                        vmax=1,
                    )
                    # sub = X_test[I, :, :, channel_i] * (1 - X_test_mask[I, :, :, channel_i])
                    # plot masked image
                    sns.heatmap(
                        X_test[I, :, :, channel_i]
                        * (1 - X_test_mask[I, :, :, channel_i]),
                        ax=axarr[channel_i, 1],
                        cbar=False,
                        vmin=-1,
                        vmax=1,
                    )
                    # sub_ = sub + predictions[I, :, :, channel_i]
                    # plot filled in image
                    sns.heatmap(
                        predictions[I, :, :, channel_i],
                        ax=axarr[channel_i, 2],
                        cbar=False,
                        vmin=-1,
                        vmax=1,
                    )

                f.tight_layout()
                f.savefig(f"img/preds.{epoch}.png", dpi=200)

    # predictions = model.predict(X_test)

    # loss_dist = tf.reduce_sum(
    #     measure_recon_loss(
    #         X_test,
    #         predictions,
    #         X_test_mask,
    #         reduction="none",
    #     ),
    #     axis=(1, 2),
    # )
    # loss = []

    # f, ax = plt.subplots()

    # for label in np.unique(y_test):
    #     label_idxs = np.where(y_test == label)[0]
    #     for i in label_idxs:
    #         loss.append({"label": label, "loss": loss_dist.numpy()[i]})
    # loss = pd.DataFrame(loss)
    # sns.boxplot(data=loss, x="label", y="loss", ax=ax)
    # sns.stripplot(data=loss, x="label", y="loss", ax=ax)
    # f.savefig("dist.png", dpi=200)

    # f, ax = plt.subplots()
    # fpr, tpr, thresholds = roc_curve(y_test, loss_dist.numpy())
    # f, ax = plt.subplots()
    # ax.plot(fpr, tpr)
    # f.savefig("roc.png")

    res = pd.DataFrame(res)

    g = sns.FacetGrid(data=res, col="loss_kind", sharey=False)
    g.map(sns.lineplot, "epoch", "loss", "group")
    g.add_legend()
    f.tight_layout()
    g.savefig("loss.png", dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-run_sweep", action="store_true")
    p.add_argument("-search_size", type=int, default=20)
    args = p.parse_args()
    main()
