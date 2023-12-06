import tensorflow as tf
from typing import Tuple
import global_vars

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model

# figure out how many filters we need to get the
# number of snps down to a width of 4

final_width = 4
n_filters = 0
while global_vars.NUM_SNPS / (2**n_filters) > final_width:
    n_filters += 1

FILTER_SIZES = [64]
for fi in range(1, n_filters):
    if fi == 1:
        FILTER_SIZES.append(FILTER_SIZES[0])
    else:
        FILTER_SIZES.append(FILTER_SIZES[fi - 1] * 2)

# define filter sizes for each of the n - 1 convolutional layers
FILTER_SIZES = [f // 1 for f in FILTER_SIZES]


def convolution(
    n_filters: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    apply_batchnorm: bool = True,
    padding: str = "same",
):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(
            n_filters,
            kernel,
            strides=stride,
            padding=padding,
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU(0.2))

    return result


def upconvolution(
    n_filters: int,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    apply_batchnorm: bool = True,
    padding: str = "same",
):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(
            n_filters,
            kernel,
            strides=stride,
            padding=padding,
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.LeakyReLU(0.2))

    return result


class ContextEncoder(Model):
    def __init__(
        self,
        kernel: Tuple[int, int] = (2, 2),
        stride: Tuple[int, int] = (2, 2),
        latent_dimensions: int = 128,
    ):
        super(ContextEncoder, self).__init__()

        self.kernel = kernel
        self.stride = stride
        self.latent_dimensions = latent_dimensions

        # self.inputs = layers.Input(
        #     shape=(
        #         global_vars.NUM_HAPLOTYPES,
        #         global_vars.NUM_SNPS,
        #         global_vars.INPUT_CHANNELS,
        #     )
        # )

        self.encoder_stack = [
            convolution(64, kernel, stride),
            convolution(64, kernel, stride),
            convolution(64, kernel, stride),
            convolution(64, kernel, stride),
        ]

        self.decoder_stack = [
            upconvolution(64, kernel, stride),
            upconvolution(64, kernel, stride),
            upconvolution(64, kernel, stride),
        ]

        # convolution(64, kernel, stride)]

    def call(self, x):
        initializer = tf.random_normal_initializer(0.0, 0.02)

        final_conv = tf.keras.layers.Conv2DTranspose(
            global_vars.OUTPUT_CHANNELS,
            self.kernel,
            strides=self.stride,
            padding="same",
            kernel_initializer=initializer,
            activation="tanh",
        )


        # Downsampling through the model
        skips = []
        for conv in self.encoder_stack:
            x = conv(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for upconv, skip in zip(self.decoder_stack, skips):
            x = upconv(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = final_conv(x)

        return x

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
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.OUTPUT_CHANNELS,
        ),
        kernel_size: int = 4,
    ):
        super(Discriminator, self).__init__()

        disc_ = [layers.Input(shape=input_shape)]

        self.convolution = layers.Conv2D(
            filter_size,
            (kernel_size, kernel_size),
            strides=(2, 2),
            padding="same",
            activation=layers.LeakyReLU(0.2),
        )

        self.final_convolution = layers.Conv2D(
            1,
            (kernel_size, kernel_size),
            strides=(1, 1),
            padding="valid",
            activation="sigmoid",
        )
        self.batch_norm = layers.BatchNormalization()

    def call(self, x):
        x = self.convolution(x)

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


if __name__ == "__main__":
    input_shape = (
        1,
        global_vars.NUM_HAPLOTYPES,
        global_vars.NUM_SNPS,
        global_vars.INPUT_CHANNELS,
    )

    generator = ContextEncoder()
    generator.build_graph(input_shape)
