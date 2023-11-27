import tensorflow as tf
from typing import Tuple
import global_vars

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model


# define filter sizes for each of the n - 1 convolutional layers
FILTER_SIZES = [64, 64, 128, 256, 512]
FILTER_SIZES = [f // 4 for f in FILTER_SIZES]


class ContextEncoderOneHot(Model):
    def __init__(
        self,
        input_shape: Tuple[int] = (
            4,
            global_vars.NUM_SNPS,
            3,
        ),
        kernel_size: int = 4,
        latent_dimensions: int = 1_000,
        activation: str = "relu",
        padding: str = "same",
    ):
        super(ContextEncoderOneHot, self).__init__()

        image_width = input_shape[1]

        n = int(np.log2(image_width))
        ni = n - 3

        # build encoder architecture
        encoder_ = [layers.Input(shape=input_shape)]

        filter_sizes = FILTER_SIZES[:ni + 1]

        for filter_size in filter_sizes:
            # block is a convolution + maxpool
            block = [
                layers.Conv2D(
                    filter_size,
                    (1, kernel_size),
                    strides=(1, 2),
                    activation=activation,
                    padding="same",
                ),
            ]
            encoder_.extend(block)
            # adjust width for max pooling operation
            image_width /= 2

        # final layer should take (4, 4, 256) and collapse to (1, 1, latent_dims)
        encoder_.append(
            layers.Conv2D(
                latent_dimensions,
                (4, kernel_size),
                strides=(1, 1),
                activation=activation,
                padding="valid",
            )
        )

        self.encoder = tf.keras.Sequential(encoder_)

        decoder_ = []

        decoder_.append(
            layers.Conv2DTranspose(
                filter_sizes[-1],
                (4, kernel_size),
                activation=activation,
                padding="valid",
                input_shape=(1, 1, latent_dimensions),
            ),
        )

        # loop over filter sizes in reverse, skipping final
        # layer.
        for filter_size in filter_sizes[1:-1][::-1]:
            # block is a convolution + maxpool
            block = [
                layers.Conv2DTranspose(
                    filter_size,
                    (1, kernel_size),
                    strides=(1, 2),
                    activation=activation,
                    padding="same",
                ),
            ]
            decoder_.extend(block)

        # final shape should be the size of the "removed" region
        # of the original image
        decoder_.append(
            layers.Conv2DTranspose(
                input_shape[-1],
                (1, kernel_size),
                strides=(1, 2),
                activation="tanh",
                padding="same",
            )
        )

        # decoder_.append(layers.Reshape((input_shape)))

        self.decoder = tf.keras.Sequential(decoder_)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


class DiscriminatorOneHot(Model):
    def __init__(
        self,
        input_shape: Tuple[int] = (
            4,
            global_vars.NUM_SNPS // 2,
            3,
        ),
        activation: str = "relu",
        kernel_size: int = 4,
    ):
        super(DiscriminatorOneHot, self).__init__()

        n = int(np.log2(input_shape[1]))
        ni = n - 3

        filter_sizes = FILTER_SIZES[:ni + 1]

        disc_ = [layers.Input(shape=input_shape)]

        for filter_size in filter_sizes:
            disc_.append(
                layers.Conv2D(
                    filter_size,
                    (1, kernel_size),
                    strides=(1, 2),
                    padding="same",
                    activation=activation,
                )
            )

        # final layer collapses to a single value
        disc_.append(
            layers.Conv2D(
                1,
                (4, kernel_size),
                strides=(1, 1),
                padding="valid",
            )
        )
        disc_.append(layers.Flatten())

        self.discriminator = tf.keras.Sequential(disc_)

    def call(self, x):
        return self.discriminator(x)

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


class ContextEncoder(Model):
    def __init__(
        self,
        input_shape: Tuple[int] = (
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            7,
        ),
        kernel_size: int = 4,
        latent_dimensions: int = 1_000,
        activation: str = "relu",
        padding: str = "same",
    ):
        super(ContextEncoder, self).__init__()

        image_width = input_shape[1]

        n = int(np.log2(image_width))
        ni = n - 3

        # build encoder architecture
        encoder_ = [layers.Input(shape=input_shape)]

        filter_sizes = FILTER_SIZES[:ni + 1]

        for filter_size in filter_sizes:
            # block is a convolution + maxpool
            block = [
                layers.Conv2D(
                    filter_size,
                    (kernel_size, kernel_size),
                    strides=(2, 2),
                    activation=activation,
                    padding="same",
                ),
            ]
            encoder_.extend(block)
            # adjust width for max pooling operation
            # image_width /= 2

        # final layer should take (4, 4, 256) and collapse to (1, 1, latent_dims)
        encoder_.append(
            layers.Conv2D(
                latent_dimensions,
                (kernel_size, kernel_size),
                strides=(1, 1),
                activation=activation,
                padding="valid",
            )
        )

        self.encoder = tf.keras.Sequential(encoder_)

        decoder_ = []

        decoder_.append(
            layers.Conv2DTranspose(
                filter_sizes[-1],
                (kernel_size, kernel_size),
                activation=activation,
                #padding="same",
                input_shape=(1, 1, latent_dimensions),
            ),
        )

        # loop over filter sizes in reverse, skipping final
        # layer.
        for filter_size in filter_sizes[1:-1][::-1]:
            # block is a convolution + maxpool
            block = [
                layers.Conv2DTranspose(
                    filter_size,
                    (kernel_size, kernel_size),
                    strides=(2, 2),
                    activation=activation,
                    padding="same",
                ),
            ]
            decoder_.extend(block)

        # final shape should be the size of the "removed" region
        # of the original image
        decoder_.append(
            layers.Conv2DTranspose(
                input_shape[-1],
                (kernel_size, kernel_size),
                strides=(2, 2),
                activation="tanh",
                padding="same",
            )
        )

        # decoder_.append(layers.Reshape((input_shape)))

        self.decoder = tf.keras.Sequential(decoder_)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


class Discriminator(Model):
    def __init__(
        self,
        input_shape: Tuple[int] = (
            global_vars.NUM_HAPLOTYPES // 2,
            global_vars.NUM_SNPS // 2,
            7,
        ),
        activation: str = "relu",
        kernel_size: int = 4,
    ):
        super(Discriminator, self).__init__()

        n = int(np.log2(input_shape[1]))
        ni = n - 3

        filter_sizes = FILTER_SIZES[:ni + 1]

        disc_ = [layers.Input(shape=input_shape)]

        for filter_size in filter_sizes:
            disc_.append(
                layers.Conv2D(
                    filter_size,
                    (kernel_size, kernel_size),
                    strides=(2, 2),
                    padding="same",
                    activation=activation,
                )
            )

        # final layer collapses to a single value
        disc_.append(
            layers.Conv2D(
                1,
                (kernel_size, kernel_size),
                strides=(1, 1),
                padding="valid",
            )
        )
        disc_.append(layers.Flatten())

        self.discriminator = tf.keras.Sequential(disc_)

    def call(self, x):
        return self.discriminator(x)

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)
