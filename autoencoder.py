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
        encoder_ = [
            layers.Input(shape=input_shape)
        ]
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
        print (nobatch)
        self.build(test_shape) # make sure to call on shape with batch
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
    

class CVAE1D(Model):
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
        latent_dimensions: int = 2,
    ):
        super(CVAE1D, self).__init__()

        self.shape = input_shape
        print (self.shape)

        self.initial_filters = initial_filters
        self.conv_layers = conv_layers
        self.conv_operations = conv_operations
        self.conv_layer_multiplier = conv_layer_multiplier
        self.kernel_size = kernel_size
        self.activation = activation
        self.fc_layers = fc_layers
        self.fc_layer_size = fc_layer_size
        self.dropout = dropout
        self.latent_dimensions = latent_dimensions

        self.padding = "valid"

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_squared_error(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def build_encoder(self):

        encoder_inputs = tf.keras.Input(shape=self.shape)

        filter_size = self.initial_filters
        image_width = self.shape[1]

        ### ENCODER
        for conv_block in range(self.conv_layers):
            # in each convolutional layer, we apply N convolutions
            # followed by a max pool operation
            for conv_op in range(self.conv_operations):
                if conv_block == 0 and conv_op == 0:
                    x = layers.Conv2D(
                            filter_size,
                            (1, self.kernel_size),
                            activation=self.activation,
                            padding=self.padding,
                        )(encoder_inputs)
                else:
                    x = layers.Conv2D(
                            filter_size,
                            (1, self.kernel_size),
                            activation=self.activation,
                            padding=self.padding,
                        )(x)
                image_width -= (self.kernel_size - 1)

            x = layers.MaxPooling2D((1, 2), padding=self.padding)(x)
            image_width /= 2
            # at the end of the block, we increase the filter size
            if conv_block + 1 < self.conv_layers:
                filter_size *= self.conv_layer_multiplier

        self.final_image_width = int(image_width)
        self.final_filter_size = int(filter_size)
        
        x = layers.Flatten()(x)

        for fc_layer in range(self.fc_layers):
            x = layers.Dense(self.fc_layer_size)(x)
            x = layers.Dropout(self.dropout)(x)

        z_mean = layers.Dense(self.latent_dimensions, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dimensions, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        
        latent_inputs = tf.keras.Input(shape=(self.latent_dimensions, ))

        for fc_layer in range(self.fc_layers):
            x = layers.Dense(self.fc_layer_size, activation=self.activation)(latent_inputs)

        x = layers.Dense(
                    self.shape[0] * self.final_image_width * self.final_filter_size,
                    activation=self.activation,
                )(x)
        x = layers.Reshape((self.shape[0], self.final_image_width, self.final_filter_size))(x)


        filter_size = self.final_filter_size
        for conv_block in range(self.conv_layers):
            # in each convolutional block, we first upsample the
            # tensor. NOTE: if this is the final convolution block,
            # our final filter size must be equal to the number of input channels.
            # if this is the final convolution block, our activation function must
            # also be tanh and not whatever activation is normally.
            final_conv_block = conv_block + 1 == self.conv_layers
            x = layers.UpSampling2D((1, 2))(x)
            for conv_op in range(self.conv_operations):
                # if this is the final convolutional operation in the block,
                # we need to reduce the filter size.
                if conv_op + 1 == self.conv_operations:
                    filter_size /= self.conv_layer_multiplier
                    x = layers.Conv2DTranspose(
                            global_vars.NUM_CHANNELS if final_conv_block else filter_size,
                            (1, self.kernel_size),
                            activation="tanh" if final_conv_block else self.activation,
                            padding=self.padding,
                        )(x)
                # otherwise, we just do a regular convolutional operation
                else:
                    x = layers.Conv2DTranspose(
                            filter_size,
                            (1, self.kernel_size),
                            activation=self.activation,
                            padding=self.padding,
                        )(x)

        return tf.keras.Model(latent_inputs, x, name="decoder")

    # def call(self, x):
    #     encoded = self.encoder(x)
    #     decoded = self.decoder(encoded)
    #     return decoded

    # def build_graph(self, test_shape):
    #     """This is for testing, based on TF tutorials"""
    #     nobatch = test_shape[1:]
    #     self.build(test_shape) # make sure to call on shape with batch
    #     gt_inputs = tf.keras.Input(shape=nobatch)

    #     if not hasattr(self, 'call'):
    #         raise AttributeError("User should define 'call' method!")

    #     _ = self.call(gt_inputs)

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
        encoder_ = [
            layers.Input(shape=input_shape)
        ]
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
                encoder_.append(layers.Dense(fc_layer_size, activation=activation))
                encoder_.append(layers.Dropout(dropout))

            encoder_.append(layers.Dense(latent_dimensions, activation=activation))

        self.encoder = tf.keras.Sequential(encoder_)

        ### DECODER
        decoder_ = []

        if latent_dimensions > 0:
            decoder_.append(layers.Input(latent_dimensions))

            for fc_layer in range(fc_layers):
                decoder_.append(layers.Dense(fc_layer_size, activation=activation))

            decoder_.append(
                    layers.Dense(
                        input_shape[0] * image_width * filter_size,
                        activation=activation,
                    ))
            decoder_.append(layers.Reshape((input_shape[0], int(image_width), filter_size)))


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
                            global_vars.NUM_CHANNELS if final_conv_block else filter_size,
                            (1, kernel_size),
                            activation="tanh" if final_conv_block else activation,
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
        self.build(test_shape) # make sure to call on shape with batch
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
        encoder_ = [
            layers.Input(shape=input_shape)
        ]
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
                encoder_.append(layers.Dense(fc_layer_size, activation=activation))
                if batch_norm:
                    encoder_.append(layers.BatchNormalization())

            encoder_.append(layers.Dense(latent_dimensions, activation=activation))

        self.encoder = tf.keras.Sequential(encoder_)

        ### DECODER
        decoder_ = []

        if latent_dimensions > 0:
            decoder_.append(layers.Input(latent_dimensions))

            for fc_layer in range(fc_layers):
                decoder_.append(layers.Dense(fc_layer_size, activation=activation))

            final_height = int(input_shape[0] / (2 ** conv_layers))
            final_width = int(input_shape[1] / (2 ** conv_layers))

            decoder_.append(
                    layers.Dense(
                        final_height * final_width * filter_size,
                        activation=activation,
                    ))
            decoder_.append(layers.Reshape((final_height, final_width, filter_size)))


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
                            global_vars.NUM_CHANNELS if final_conv_block else filter_size,
                            (kernel_size, kernel_size),
                            activation="tanh" if final_conv_block else activation,
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
        self.build(test_shape) # make sure to call on shape with batch
        gt_inputs = tf.keras.Input(shape=nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method!")

        _ = self.call(gt_inputs)