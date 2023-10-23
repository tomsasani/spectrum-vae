import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple
import global_vars
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class CNNRiley(Model):

    def __init__(
        self,
        input_shape: Tuple[int] = (
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        ),
        activation: str = "relu",
    ):
        super(CNNRiley, self).__init__()

        # build encoder architecture
        self.model_ = [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (1, 5), activation=activation),
            layers.MaxPooling2D((1, 2)),
            layers.Conv2D(64, (1, 5), activation=activation),
            layers.MaxPooling2D((1, 2)),
            layers.Lambda(lambda x: tf.reduce_sum(x, axis=1)),
            layers.Flatten(),
            layers.Dense(128, activation=activation),
            layers.Dense(128, activation=activation),
            layers.Dense(1),
        ]

    def call(self, x):
        m = tf.keras.Sequential(self.model_)
        return m(x)

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


class CNN1D(Model):

    def __init__(
        self,
        input_shape: Tuple[int] = (
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        ),
        initial_filters: int = 8,
        conv_layer_multiplier: int = 2,
        conv_layers: int = 3,
        fc_layers: int = 2,
        fc_layer_size: int = 64,
        dropout: float = 0.5,
        activation: str = "elu",
        kernel_size: int = 3,
        conv_operations: int = 2,
    ):
        super(CNN1D, self).__init__()

        # build encoder architecture
        model_ = [layers.Input(shape=input_shape)]
        padding = "valid"
        filter_size = initial_filters
        image_width = input_shape[1]

        ### ENCODER
        for conv_block in range(conv_layers):
            # in each convolutional layer, we apply N convolutions
            # followed by a max pool operation
            for conv_op in range(conv_operations):
                model_.append(
                    layers.Conv2D(
                        filter_size,
                        (1, kernel_size),
                        activation=activation,
                        padding=padding,
                    ))
            model_.append(layers.MaxPooling2D((1, 2), padding=padding))

            # at the end of the block, we increase the filter size
            if conv_block + 1 < conv_layers:
                filter_size *= conv_layer_multiplier

        model_.append(layers.Flatten())

        for fc_layer in range(fc_layers):
            model_.append(layers.Dense(fc_layer_size, activation=activation))
            model_.append(layers.Dropout(dropout))

        model_.append(layers.Dense(1, activation=activation))

        self.model_ = model_

    def call(self, x):
        m = tf.keras.Sequential(self.model_)
        return m(x)

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)

class CNNPadded(Model):

    def __init__(
        self,
        input_shape: Tuple[int] = (
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        ),
        initial_filters: int = 8,
        conv_layer_multiplier: int = 2,
        conv_layers: int = 3,
        fc_layers: int = 2,
        fc_layer_size: int = 64,
        dropout: float = 0.5,
        activation: str = "elu",
        kernel_size: int = 3,
        conv_operations: int = 2,
    ):
        super(CNNPadded, self).__init__()

        # build encoder architecture
        model_ = [layers.Input(shape=input_shape)]
        padding = "same"
        filter_size = initial_filters

        ### ENCODER
        for conv_block in range(conv_layers):
            # in each convolutional layer, we apply N convolutions
            # followed by a max pool operation
            for conv_op in range(conv_operations):
                model_.append(
                    layers.Conv2D(
                        filter_size,
                        (kernel_size, kernel_size),
                        activation=activation,
                        padding=padding,
                    ))
            model_.append(layers.MaxPooling2D((2, 2), padding=padding))

            # at the end of the block, we increase the filter size
            if conv_block + 1 < conv_layers:
                filter_size *= conv_layer_multiplier

        model_.append(layers.Flatten())

        for fc_layer in range(fc_layers):
            model_.append(layers.Dense(fc_layer_size, activation=activation))
            model_.append(layers.Dropout(dropout))

        model_.append(layers.Dense(1, activation=activation))

        self.model_ = model_

    def call(self, x):
        m = tf.keras.Sequential(self.model_)
        return m(x)

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


class AE(Model):

    def __init__(
        self,
        input_shape: Tuple[int] = (
            # 1,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        ),
        fc_layers: int = 3,
        fc_layer_size: int = 128,
        dropout: float = 0.5,
        activation: str = "elu",
        latent_dimensions: int = 8,
    ):
        super(AE, self).__init__()

        # build encoder architecture
        encoder_ = [layers.Input(shape=input_shape)]
        encoder_.append(layers.Flatten())

        ### ENCODER
        for fc_layer in range(fc_layers):
            encoder_.append(layers.Dense(fc_layer_size, activation=activation))
            encoder_.append(layers.Dropout(dropout))

        encoder_.append(layers.Dense(latent_dimensions, activation=activation))

        self.encoder = tf.keras.Sequential(encoder_)

        ### DECODER
        decoder_ = []

        decoder_.append(layers.Input(shape=latent_dimensions))

        for fc_layer in range(fc_layers):
            decoder_.append(layers.Dense(fc_layer_size, activation=activation))

        decoder_.append(
            layers.Dense(
                input_shape[0] * input_shape[1],
                activation="tanh",
            ))
        decoder_.append(layers.Reshape((input_shape)))

        self.decoder = tf.keras.Sequential(decoder_)

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        print(nobatch)
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class CAE1D(Model):

    def __init__(
        self,
        input_shape: Tuple[int] = (
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        ),
        initial_filters: int = 8,
        conv_layer_multiplier: int = 2,
        conv_layers: int = 3,
        fc_layers: int = 2,
        fc_layer_size: int = 64,
        dropout: float = 0.5,
        activation: str = "elu",
        kernel_size: int = 3,
        conv_operations: int = 2,
        latent_dimensions: int = 8,
    ):
        super(CAE1D, self).__init__()

        # build encoder architecture
        encoder_ = [layers.Input(shape=input_shape)]
        padding = "valid"

        filter_size = initial_filters

        image_width = input_shape[1]

        ### ENCODER
        for conv_block in range(conv_layers):
            # in each convolutional layer, we apply N convolutions
            # followed by a max pool operation
            for conv_op in range(conv_operations):
                encoder_.append(
                    layers.Conv2D(
                        filter_size,
                        (1, kernel_size),
                        activation=activation,
                        padding=padding,
                    ))
                image_width -= (kernel_size - 1)
                encoder_.append(layers.BatchNormalization())
            encoder_.append(layers.MaxPooling2D((1, 2), padding=padding))
            image_width /= 2
            # at the end of the block, we increase the filter size
            if conv_block + 1 < conv_layers:
                filter_size *= conv_layer_multiplier

        if latent_dimensions > 0:
            encoder_.append(layers.Flatten())

            for fc_layer in range(fc_layers):
                encoder_.append(
                    layers.Dense(fc_layer_size, activation=activation))
                encoder_.append(layers.Dropout(dropout))

            encoder_.append(
                layers.Dense(latent_dimensions, activation=activation))

        self.encoder = tf.keras.Sequential(encoder_)

        ### DECODER
        decoder_ = []

        if latent_dimensions > 0:
            decoder_.append(layers.Input(latent_dimensions))

            for fc_layer in range(fc_layers):
                decoder_.append(
                    layers.Dense(fc_layer_size, activation=activation))

            decoder_.append(
                layers.Dense(
                    input_shape[0] * image_width * filter_size,
                    activation=activation,
                ))
            decoder_.append(
                layers.Reshape(
                    (input_shape[0], int(image_width), filter_size)))

        for conv_block in range(conv_layers):
            # in each convolutional block, we first upsample the
            # tensor. NOTE: if this is the final convolution block,
            # our final filter size must be equal to the number of input channels.
            # if this is the final convolution block, our activation function must
            # also be tanh and not whatever activation is normally.
            final_conv_block = conv_block + 1 == conv_layers
            decoder_.append(layers.UpSampling2D((1, 2)))
            for conv_op in range(conv_operations):
                # if this is the final convolutional operation in the block,
                # we need to reduce the filter size.
                if conv_op + 1 == conv_operations:
                    filter_size /= conv_layer_multiplier
                    decoder_.append(
                        layers.Conv2DTranspose(
                            global_vars.NUM_CHANNELS
                            if final_conv_block else filter_size,
                            (1, kernel_size),
                            activation="tanh"
                            if final_conv_block else activation,
                            padding=padding,
                        ))
                # otherwise, we just do a regular convolutional operation
                else:
                    decoder_.append(
                        layers.Conv2DTranspose(
                            filter_size,
                            (1, kernel_size),
                            activation=activation,
                            padding=padding,
                        ))

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

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)


class CAEPadded(Model):

    def __init__(
        self,
        input_shape: Tuple[int] = (
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        ),
        initial_filters: int = 8,
        conv_layer_multiplier: int = 2,
        conv_layers: int = 3,
        fc_layers: int = 2,
        fc_layer_size: int = 64,
        dropout: float = 0.5,
        activation: str = "elu",
        kernel_size: int = 3,
        conv_operations: int = 2,
        latent_dimensions: int = 8,
        batch_norm: bool = False,
    ):
        super(CAEPadded, self).__init__()

        # build encoder architecture
        encoder_ = [layers.Input(shape=input_shape)]
        padding = "same"

        filter_size = initial_filters

        ### ENCODER
        for conv_block in range(conv_layers):
            # in each convolutional layer, we apply N convolutions
            # followed by a max pool operation
            for conv_op in range(conv_operations):
                encoder_.append(
                    layers.Conv2D(
                        filter_size,
                        (kernel_size, kernel_size),
                        activation=activation,
                        padding=padding,
                    ))
                if batch_norm:
                    encoder_.append(layers.BatchNormalization())
            encoder_.append(layers.MaxPooling2D((2, 2), padding=padding))
            # adjust height
            # cur_height //= 2
            # if cur_height % 2 != 0 and conv_block + 1 < conv_layers:
            #     encoder_.append(layers.ZeroPadding2D(((0, 1), (0, 0))))
            #     cur_height += 1
            # at the end of the block, we increase the filter size
            if conv_block + 1 < conv_layers:
                filter_size *= conv_layer_multiplier

        if latent_dimensions > 0:
            encoder_.append(layers.Flatten())

            for fc_layer in range(fc_layers):
                encoder_.append(
                    layers.Dense(fc_layer_size, activation=activation))
                if batch_norm:
                    encoder_.append(layers.BatchNormalization())

            encoder_.append(
                layers.Dense(latent_dimensions, activation=activation))

        self.encoder = tf.keras.Sequential(encoder_)

        ### DECODER
        decoder_ = []

        if latent_dimensions > 0:
            decoder_.append(layers.Input(latent_dimensions))

            for fc_layer in range(fc_layers):
                decoder_.append(
                    layers.Dense(fc_layer_size, activation=activation))

            final_height = int(input_shape[0] / (2**conv_layers))
            final_width = int(input_shape[1] / (2**conv_layers))

            decoder_.append(
                layers.Dense(
                    final_height * final_width * filter_size,
                    activation=activation,
                ))
            decoder_.append(
                layers.Reshape((final_height, final_width, filter_size)))

        for conv_block in range(conv_layers):
            # in each convolutional block, we first upsample the
            # tensor. NOTE: if this is the final convolution block,
            # our final filter size must be equal to the number of input channels.
            # if this is the final convolution block, our activation function must
            # also be tanh and not whatever activation is normally.
            final_conv_block = conv_block + 1 == conv_layers
            decoder_.append(layers.UpSampling2D((2, 2)))
            for conv_op in range(conv_operations):
                # if this is the final convolutional operation in the block,
                # we need to reduce the filter size.
                if conv_op + 1 == conv_operations:
                    filter_size /= conv_layer_multiplier
                    decoder_.append(
                        layers.Conv2D(
                            global_vars.NUM_CHANNELS
                            if final_conv_block else filter_size,
                            (kernel_size, kernel_size),
                            activation="tanh"
                            if final_conv_block else activation,
                            padding=padding,
                        ))
                # otherwise, we just do a regular convolutional operation
                else:
                    decoder_.append(
                        layers.Conv2D(
                            filter_size,
                            (kernel_size, kernel_size),
                            activation=activation,
                            padding=padding,
                        ))

        self.decoder = tf.keras.Sequential(decoder_)

    def call(self, x):
        #image = tf.expand_dims(x[:, :, :, 0], axis=3)
        #dists = x[:, 0, :, 1]
        encoded = self.encoder(x)
        #encoded_ = tf.concat((encoded, dists), 1)
        decoded = self.decoder(encoded)
        return decoded

    def build_graph(self, test_shape):
        """This is for testing, based on TF tutorials"""
        nobatch = test_shape[1:]
        self.build(test_shape)  # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)