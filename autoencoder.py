import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import global_vars

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class CAE(Model):
    def __init__(
        self,
        initial_filters: int = 8,
        conv_layers: int = 2,
        fc_layers: int = 2,
        latent_dimensions: int = 16,
        fc_layer_size: int = 64,
        dropout: float = 0.2,
    ):
        super(CAE, self).__init__()
        self.latent_dim = latent_dimensions

        input_shape = (
            global_vars.NUM_HAPLOTYPES,
            global_vars.NUM_SNPS,
            global_vars.NUM_CHANNELS,
        )

        # build encoder architecture
        encoder_ = [layers.Input(shape=input_shape)]
        conv_filters = initial_filters
        for _ in range(conv_layers):
            encoder_.append(layers.Conv2D(conv_filters, (1, 5), activation="relu"))
            encoder_.append(layers.MaxPooling2D((1, 2)))
            conv_filters *= 2

        encoder_.append(layers.Flatten())
        for _ in range(fc_layers):
            encoder_.append(layers.Dense(fc_layer_size))
            encoder_.append(layers.Dropout(dropout))
        encoder_.append(layers.Dense(latent_dimensions))

        self.encoder = tf.keras.Sequential(encoder_)

        # figure out the final number of filters used in the encoder
        # NOTE: we divide by two since the number of filters gets multiplied
        # at the end of every iteration in the loop above.
        conv_filters = int(conv_filters / 2)

        # build decoder architecture
        decoder_ = [layers.Input(shape=latent_dimensions)]
        for _ in range(fc_layers):
            decoder_.append(layers.Dense(fc_layer_size))

        # figure out size of encoder output
        final_width = global_vars.NUM_SNPS
        for _ in range(conv_layers):
            final_width -= 4 # 1 x 5 conv
            final_width /= 2 # 1 x 2 pool
        final_width = int(final_width)

        # figure out final number of conv filters
        final_filter = int(conv_filters / 2)

        decoder_.append(layers.Dense(global_vars.NUM_HAPLOTYPES * final_width * conv_filters))
        decoder_.append(layers.Reshape((global_vars.NUM_HAPLOTYPES, final_width, conv_filters)))

        for _ in range(conv_layers - 1):
            decoder_.append(layers.UpSampling2D((1, 2)))
            decoder_.append(layers.Conv2DTranspose(final_filter, (1, 5), activation='relu'))
            final_filter /= 2
        # final upsampling and Conv2D layers
        decoder_.append(layers.UpSampling2D((1, 2)))
        decoder_.append(layers.Conv2DTranspose(global_vars.NUM_CHANNELS, kernel_size=(1, 5), activation='tanh'))

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

class CAEManual(Model):
    def __init__(self, latent_dimensions: int = 8):
        super(CAEManual, self).__init__()
        self.latent_dim = latent_dimensions

        input_shape = (
            global_vars.NUM_HAPLOTYPES, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS,
        )

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(8, (1, 5), activation='relu'), # output is (100, 32, 16)
            layers.MaxPooling2D((1, 2)),
            layers.Conv2D(16, (1, 5), activation='relu'), # output is (100, 28, 32)
            layers.MaxPooling2D((1, 2)),
            layers.Flatten(),
            layers.Dense(latent_dimensions),
            
        ])


        self.decoder = tf.keras.Sequential([
            layers.Input(shape=latent_dimensions),
            layers.Dense(50 * 6 * 16),
            layers.Reshape((50, 6, 16)),
            #layers.UpSampling2D((1, 2)),
            #layers.Conv2DTranspose(16, (1, 5), activation='relu', padding='valid'),
            layers.UpSampling2D((1, 2),),
            layers.Conv2DTranspose(8, (1, 5), activation='relu'),
            layers.UpSampling2D((1, 2),),
            layers.Conv2DTranspose(1, (1, 5), activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        print (encoded.shape)
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
