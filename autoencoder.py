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


class CVAE(Model):
    def __init__(self, latent_dim: int = 16):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        input_shape = (global_vars.NUM_HAPLOTYPES, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS)

        # [ (W - K + 2P) / S ] + 1

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(8, (1, 5), activation='relu'), # output is (100, 32, 16)
            layers.Conv2D(16, (1, 5), activation='relu'), # output is (100, 28, 32)
            layers.Flatten(), # output is 100 * 28 * 32
            layers.Dense(latent_dim + latent_dim),
        ])


        self.decoder = tf.keras.Sequential([
            layers.Input(shape = (latent_dim, )),
            layers.Dense(global_vars.NUM_HAPLOTYPES * 28 * 16),
            layers.Reshape((global_vars.NUM_HAPLOTYPES, 28, 16)),
            layers.Conv2DTranspose(16, (1, 5), activation='relu'),
            #layers.UpSampling2D((1, 2)),
            layers.Conv2DTranspose(8, (1, 5), activation='relu'),
            #layers.UpSampling2D((1, 2)),
            layers.Conv2D(global_vars.NUM_CHANNELS, kernel_size=(1, 1), activation='tanh'),
        ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    

class CAE(Model):
    def __init__(self, latent_dim: int = 64, kernel_size: int = 5):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim

        input_shape = (global_vars.NUM_HAPLOTYPES, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS)

        # [ (W - K + 2P) / S ] + 1

        # assuming two convolutional layers of kernel_size
        first_xdim = (global_vars.NUM_SNPS - kernel_size) + 1
        first_xdim = int(( (first_xdim - 2) / 2 ) + 1)
        second_xdim = ( (first_xdim - kernel_size) ) + 1
        second_xdim = int(( (second_xdim - 2) / 2 ) + 1)

        print (first_xdim, second_xdim)

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (1, kernel_size), activation='relu'), 
            layers.MaxPooling2D(pool_size = (1,2), strides = (1,2)), 
            layers.Conv2D(32, (1, kernel_size), activation='relu'), 
            layers.MaxPooling2D(pool_size = (1,2), strides = (1,2)), 
            layers.Flatten(),
            layers.Dense(latent_dim),
        ])


        self.decoder = tf.keras.Sequential([
            layers.Input(shape = (latent_dim, )),
            layers.Dense(global_vars.NUM_HAPLOTYPES * second_xdim * 32), # 100, 6, filters
            layers.Reshape((global_vars.NUM_HAPLOTYPES, second_xdim, 32)), # 100, 6, filters
            layers.UpSampling2D((1, 2)),
            layers.Conv2DTranspose(32, (1, kernel_size), activation='relu'), # filters
            layers.UpSampling2D((1, 2)),
            layers.Conv2DTranspose(16, (1, kernel_size), activation='relu'),
            
            layers.Conv2D(global_vars.NUM_CHANNELS, kernel_size=(1, 1), activation='tanh'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
