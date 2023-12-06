import tensorflow as tf
from typing import Tuple
import global_vars

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.models import Model

# figure out how many filters we need to get the
# number of snps down to a width of 1

final_width = 1
n_filters = 0
while global_vars.NUM_SNPS / (2**n_filters) > final_width:
    n_filters += 1

FILTER_SIZES = [64]
for fi in range(1, n_filters):
    if fi >= 3:
        FILTER_SIZES.append(FILTER_SIZES[fi - 1])
    else:
        FILTER_SIZES.append(FILTER_SIZES[fi - 1] * 2)


# define filter sizes for each of the n - 1 convolutional layers
FILTER_SIZES = [f // 2 for f in FILTER_SIZES]


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU(0.2))

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(
        shape=[
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.INPUT_CHANNELS,
        ]
    )

    down_stack = [downsample(64, 4, apply_batchnorm=False)]
    for filter_size in FILTER_SIZES[1:]:
        down_stack.append(downsample(filter_size, 4))

    up_stack = []
    for fi, filter_size in enumerate(FILTER_SIZES[::-1][1:]):
        up_stack.append(
            upsample(
                filter_size,
                4,
                apply_dropout=True if fi <= 2 else False,
            )
        )

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        global_vars.OUTPUT_CHANNELS,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


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

        for filter_size in FILTER_SIZES[:-2]:
            block = [
                layers.Conv2D(
                    filter_size,
                    (kernel_size, kernel_size),
                    strides=(2, 2),
                    padding="same",
                    activation=layers.LeakyReLU(0.2),
                ),
                layers.BatchNormalization(),
            ]
            disc_.extend(block)

        final_block = [
                layers.Conv2D(
                    1,
                    (kernel_size, kernel_size),
                    strides=(1, 1),
                    padding="valid",
                    # activation="sigmoid",
                ),
            ]
        disc_.extend(final_block)

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


generator = Generator()
tf.keras.utils.plot_model(
    generator,
    show_shapes=True,
    dpi=64,
    to_file="model.png",
)
