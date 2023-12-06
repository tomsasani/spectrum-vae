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
FILTER_SIZES = [f // 1 for f in FILTER_SIZES]


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


# def Discriminator():
#     initializer = tf.random_normal_initializer(0.0, 0.02)

#     inp = tf.keras.layers.Input(shape=[global_vars.NUM_HAPLOTYPES, global_vars.NUM_SNPS, 3], name="input_image")
#     tar = tf.keras.layers.Input(shape=[global_vars.NUM_HAPLOTYPES, global_vars.NUM_SNPS, 3], name="target_image")

#     x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

#     down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
#     down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
#     down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

#     zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
#     conv = tf.keras.layers.Conv2D(
#         512, 4, strides=1, kernel_initializer=initializer, use_bias=False
#     )(
#         zero_pad1
#     )  # (batch_size, 31, 31, 512)

#     batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

#     leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

#     zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

#     last = tf.keras.layers.Conv2D(1, 5, strides=1, kernel_initializer=initializer)(
#         zero_pad2
#     )  # (batch_size, 30, 30, 1)

#     return tf.keras.Model(inputs=[inp, tar], outputs=last)


def Discriminator():

    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = layers.Input(
        shape=[
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.INPUT_CHANNELS,
        ],
        name="input_image",
    )
    tar = layers.Input(
        shape=[
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.OUTPUT_CHANNELS,
        ],
        name="target_image",
    )

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, apply_batchnorm=False)(tar)
    down2 = downsample(128, 4)(down1)
    #down3 = downsample(256, 4)(down2) # 4

    final = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), kernel_initializer=initializer)(down2)
    return tf.keras.Model(inputs=[inp, tar], outputs=final)


generator = Generator()
tf.keras.utils.plot_model(
    generator,
    show_shapes=True,
    dpi=64,
    to_file="model.png",
)
