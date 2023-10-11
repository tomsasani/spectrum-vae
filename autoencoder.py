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
    def __init__(self, latent_dim: int = 64, kernel_size: int = 5):
        super(CAE, self).__init__()
        self.latent_dim = latent_dim

        input_shape = (global_vars.NUM_HAPLOTYPES, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS,)

        # [ (W - K + 2P) / S ] + 1

        # assuming two convolutional layers of kernel_size
        first_xdim = (global_vars.NUM_SNPS - kernel_size) + 1
        #first_xdim = int(( (first_xdim - 2) / 2 ) + 1)
        second_xdim = ( (first_xdim - kernel_size) ) + 1
        #second_xdim = int(( (second_xdim - 2) / 2 ) + 1)

        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            #layers.Rescaling(1./255),
            layers.Conv2D(32, (1, 5), activation='relu'), # output is (100, 32, )
            layers.MaxPooling2D((1, 2)), # output is (100, 16, )
            layers.Conv2D(64, (1, 5), activation='relu'), # output is (100, 12, )
            layers.MaxPooling2D((1, 2)), # output is (100, 6, )
            layers.Conv2D(128, (1, 5), activation='relu'), # output is (100, 12, )
            layers.MaxPooling2D((1, 2)), # output is (100, 6, )
            layers.Flatten(), # output is 100 * 2 * 32
            layers.Dense(32),
            layers.Dropout(0.5),
            layers.Dense(32),
            layers.Dropout(0.5),
            layers.Dense(latent_dim),
        ])


        self.decoder = tf.keras.Sequential([
            layers.Input(shape = (latent_dim, )),
            layers.Dense(32),
            layers.Dense(32),
            layers.Dense(global_vars.NUM_HAPLOTYPES * 1 * 128),
            layers.Reshape((global_vars.NUM_HAPLOTYPES, 1, 128)),
            layers.UpSampling2D((1, 2)),
            layers.Conv2DTranspose(128, (1, 5), activation='relu'),
            layers.UpSampling2D((1, 2)),
            layers.Conv2DTranspose(64, (1, 5), activation='relu'),
            layers.UpSampling2D((1, 2)),
            layers.Conv2DTranspose(32, (1, 5), activation='relu'),
            layers.Conv2D(global_vars.NUM_CHANNELS, kernel_size=(1, 1), activation='sigmoid'),
        ])

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
