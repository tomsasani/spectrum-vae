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


def batch_random_mask(X: np.ndarray, pct_mask: float = 0.25):
    # get shape of array
    batch_size, n_haps, n_snps, n_channels = X.shape

    mask_arr = np.zeros(X.shape)

    for i in np.arange(batch_size):
        i_mask = create_random_image_mask(
            X[i],
            pct_mask=pct_mask,
        )
        mask_arr[i] = i_mask
    return mask_arr


def create_random_image_mask(X: np.ndarray, pct_mask: float = 0.25):
    # get shape of array
    n_haps, n_snps, n_channels = X.shape

    mask = np.zeros(X.shape)

    # define random blocks that are each 1/16 of the width of the image
    segment_width = n_snps // 8

    mask[segment_width : segment_width * 3, segment_width : segment_width * 3, :] = 1

    while (np.sum(mask[:, :, 0]) / (n_snps**2)) < pct_mask:
        # pick a random location in the image where we'll initialize
        # the "bottom left" of the segment mask.
        rand_x = np.random.randint(n_snps - segment_width)
        rand_y = np.random.randint(n_snps - segment_width)

        mask[rand_x : rand_x + segment_width, rand_y : rand_y + segment_width, :] = 1

    return mask


def create_image_mask(X: np.ndarray, pct_mask: float = 0.25):
    # get shape of array
    n_haps, n_snps, n_channels = X.shape

    # figure out dimensions of masked region
    i = int(n_haps * pct_mask)
    j = i * 3

    # create new input image
    X_ = X.copy()

    # create the array that represents the expected "middle"
    # of the image
    y_ = X[i:j, i:j, :]
    # add the "left" and "right" sides of the original image

    X_[i:j, i:j, :] = 0

    return X_, y_


def batch_mask(X: np.ndarray, pct_mask: float = 0.25):
    # get shape of array
    batch_size, n_haps, n_snps, n_channels = X.shape

    # make sure image is square
    assert n_haps == n_snps

    # figure out dimensions of masked region
    mask_dim = int(n_haps * pct_mask) * 2

    X_masked_batch = np.zeros(X.shape)
    X_true_batch = np.zeros((batch_size, mask_dim, mask_dim, n_channels))

    for i in np.arange(batch_size):
        X_masked, X_true = create_image_mask(X[i])
        X_masked_batch[i] = X_masked
        X_true_batch[i] = X_true
    return X_masked_batch, X_true_batch

def batch_sort(X: np.ndarray):
    batch_size, n_haps, n_snps, n_channels = X.shape

    X_new = np.zeros(X.shape)
    for batch_i in np.arange(batch_size):
        orig_X = X[batch_i]
        # get indices along which to sort haplotypes.
        # just use first channel without distances
        sorted_hap_idxs = util.sort_min_diff(orig_X[:, :, 0])
        orig_X_sorted = orig_X[sorted_hap_idxs, :, :]
        X_new[batch_i] = orig_X_sorted
    return X_new


def build_model(
    input_shape: Tuple[int] = (
        1,
        4,
        global_vars.NUM_SNPS,
        global_vars.NUM_CHANNELS,
    ),
    activation: str = "elu",
    latent_dimensions: int = 8,
):
    model = context_encoder.ContextEncoder(
        input_shape=input_shape,
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
    beta_1: float = 0.5,
    beta_2: float = 0.999,
    decay_rate: float = 0.96,
    decay_steps: int = 50,
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
    loss_func = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
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

    return (true_disc_loss + fake_disc_loss) / 2.0


def measure_recon_loss(
    y_true,
    y_pred,
    mask,
    reduction: str = "auto",
):
    loss_func = tf.keras.losses.MeanSquaredError(reduction=reduction)
    #return tf.reduce_mean(mask * ((y_true - y_pred) ** 2))
    return loss_func(y_true, y_pred, sample_weight=mask[:, :, :, 0] * 10)


def measure_total_loss(
    discriminator,
    y_true,
    y_pred,
    mask,
    l2_weight: float = 0.999,
    reduction: str = "auto",
):
    disc_loss = measure_disc_loss(
        discriminator,
        y_pred,
        y_true,
        reduction=reduction,
    )
    recon_loss = measure_recon_loss(
        y_true,
        y_pred,
        mask,
        reduction=reduction,
    )

    return (recon_loss * l2_weight) + (disc_loss * (1 - l2_weight))


def train_step(
    context_encoder,
    discriminator,
    ce_opt,
    disc_opt,
    X: np.ndarray,
    mask: np.ndarray,
    l2_weight: float = 0.999,
):
    with tf.GradientTape(persistent=True) as tape:
        # per eq. 1, we run the context encoder (i.e., generator) on
        # the input image X, after X is multiplied by the inverse of
        # the image mask. the original mask should contain 1s where pixels
        # were dropped and 0s where they were kept, so the inverse will contain
        # 0s at dropped pixels. in other words, we run the context encoder
        # on the original image with the masked pixels removed.

        predicted_fill = context_encoder(X * (1 - mask), training=True)

        # calculate simple L2 loss between predicted and true fill. we weight
        # the L2 loss by the dropped pixels.
        l2_loss = measure_recon_loss(X, predicted_fill, mask)
        disc_loss = measure_disc_loss(discriminator, X, predicted_fill)
        # calculate total loss that we'll use to update the context encoder
        total_loss = (l2_loss * l2_weight) + (disc_loss * (1 - l2_weight))

    # update discriminator using discriminator loss (so
    # that discriminator gets better)
    disc_grads = tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # update CE using CE loss (so that CE gets better)
    ce_grads = tape.gradient(l2_loss, context_encoder.trainable_variables)
    ce_opt.apply_gradients(zip(ce_grads, context_encoder.trainable_variables))

    return disc_loss, l2_loss, total_loss


def pad_training_data(train_data: np.ndarray):
    # pad if necessary
    padding = 0
    input_shape = train_data.shape
    cur_height, cur_width = input_shape[1:3]
    while cur_height < cur_width:
        padding += 1
        cur_height += 1
    padding_zeros = np.zeros((input_shape[0], padding, input_shape[2], input_shape[3]))
    train_data = np.concatenate((train_data, padding_zeros), axis=1)
    return train_data


def format_cifar(data: np.ndarray):
    n_examples = data.shape[0]
    new_data = np.zeros((n_examples, 32, 32, 3))
    for i in np.arange(n_examples):
        img = data[i]
        norm_img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        norm_img = (2 * norm_img) - 1
        new_data[i] = norm_img
    return new_data


def main():
    CIFAR = False

    if CIFAR:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

        N = 10_000
        X_train, y_train = format_cifar(X_train[:N]), y_train[:N]
        X_test, y_test = format_cifar(X_test[:N]), y_test[:N]
    else:
        with np.load("data/data.npz") as data:
            train_data = data["train_data"]
            train_labels = data["train_labels"]

        train_data = batch_sort(train_data)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            train_data,
            train_labels,
            test_size=0.2,
            shuffle=True,
        )

    # split out individual rows (i.e., haplotypes) into new arrays
    # train_data = np.concatenate(train_data, axis=2)
    # train_data = np.transpose(train_data, (2, 0, 1, 3))

    # train_data = pad_training_data(train_data)

    input_shape = X_train.shape

    # generate a masked version of the input, along with
    # the true values present in the mask
    # X_train_masked, X_train_true = batch_mask(X_train, pct_mask=0.25)
    # X_test_masked, X_test_true = batch_mask(X_test, pct_mask=0.25)

    # generate random masks for every image
    X_train_mask = batch_random_mask(X_train, pct_mask=0.25)
    X_test_mask = batch_random_mask(X_test, pct_mask=0.25)

    if CIFAR:
        f, axarr = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
        I = np.random.randint(X_train.shape[0])
        a = axarr[0].imshow(X_train[I], vmin=-1, vmax=1)
        b = axarr[1].imshow(X_train[I] * (1 - X_train_mask[I]), vmin=-1, vmax=1)
        f.colorbar(b, ax=axarr[1])

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
                cbar=True,
            )
            sns.heatmap(
                X_train[0, :, :, channel_i] * (1 - X_train_mask[0, :, :, channel_i]),
                ax=axarr[1, channel_i],
                cbar=True,
            )

        f.tight_layout()
        f.savefig("orig.png", dpi=200)

    model = build_model(
        input_shape=input_shape[1:],
        activation="relu",
        latent_dimensions=1_000,
    )

    disc = context_encoder.Discriminator()
    disc.build_graph(
        (
            1,
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        )
    )
    disc.discriminator.summary()

    # context encoder LR is 10x higher than discriminator
    LR = 2e-4
    ce_optimizer = build_optimizer(learning_rate=LR * 10)
    disc_optimizer = build_optimizer(learning_rate=LR)

    EPOCHS = 20
    EPOCHS += 1
    BATCH_SIZE = 64

    res = []

    for epoch in tqdm.tqdm(range(EPOCHS)):
        for step in range(X_train.shape[0] // BATCH_SIZE):
            start_idx = BATCH_SIZE * step
            end_idx = BATCH_SIZE * (step + 1)

            disc_loss, recon_loss, total_loss = train_step(
                model,
                disc,
                ce_optimizer,
                disc_optimizer,
                X_train[start_idx:end_idx],
                X_train_mask[start_idx:end_idx],
            )

        # check out the model at each step
        predictions = model.predict(X_test * (1 - X_test_mask))

        # measure test loss
        disc_loss_test = measure_disc_loss(disc, X_test, predictions)
        recon_loss_test = measure_recon_loss(X_test, predictions, X_test_mask)
        total_loss_test = measure_total_loss(disc, X_test, predictions, X_test_mask)

        print(f"Discriminator training loss: {disc_loss.numpy()}")
        print(f"L2 training loss: {recon_loss.numpy()}")

        print(f"Discriminator validation loss: {disc_loss_test.numpy()}")
        print(f"L2 validation loss: {recon_loss_test.numpy()}")

        for loss, label, group in zip(
            (
                disc_loss,
                recon_loss,
                total_loss,
                disc_loss_test,
                recon_loss_test,
                total_loss_test,
            ),
            ("disc", "recon", "total", "disc", "recon", "total"),
            (
                "training",
                "training",
                "training",
                "validation",
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
                    sub = X_test[idx, :, :, :]
                    mask = X_test_mask[idx, :, :, :]
                    axarr[0, plot_idx].imshow(sub)
                    axarr[1, plot_idx].imshow(sub * (1 - mask))
                    axarr[2, plot_idx].imshow(predictions[idx, :, :, :])

                f.tight_layout()
                f.savefig(f"img/preds.{epoch}.png", dpi=200)
            else:
                f, axarr = plt.subplots(
                    global_vars.NUM_CHANNELS, 3, figsize=(28, 15), sharey=True
                )
                I = np.random.randint(X_test.shape[0])
                for channel_i in range(global_vars.NUM_CHANNELS):
                    # plot original image
                    sub = X_test[I, :, :, channel_i]
                    mask = X_test_mask[I, :, :, channel_i]
                    sns.heatmap(
                        sub,
                        ax=axarr[channel_i, 0],
                        cbar=True,
                        vmin=-1,
                        vmax=1,
                    )
                    # plot masked image
                    sns.heatmap(
                        sub * (1 - mask),
                        ax=axarr[channel_i, 1],
                        cbar=True,
                        vmin=-1,
                        vmax=1,
                    )
                    # plot filled in image
                    
                    sns.heatmap(
                        predictions[I, :, :, channel_i],
                        ax=axarr[channel_i, 2],
                        cbar=True,
                        vmin=-1,
                        vmax=1,
                    )

                f.tight_layout()
                f.savefig(f"img/preds.{epoch}.png", dpi=200)

    predictions = model.predict(X_test * (1 - X_test_mask))

    loss_dist = tf.reduce_mean(
        measure_recon_loss(
            X_test,
            predictions,
            X_test_mask,
            reduction="none",
        ),
        axis=(1, 2),
    )
    loss = []

    # f, ax = plt.subplots()

    # for label in np.unique(y_test):
    #     label_idxs = np.where(y_test == label)[0]
    #     for i in label_idxs:
    #         loss.append({"label": str(label), "loss": loss_dist.numpy()[i]})
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
