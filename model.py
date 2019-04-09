#-*- coding: utf-8 -*-
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU,\
    Conv2DTranspose,Dense, ZeroPadding2D,Reshape,concatenate,Lambda, Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
# setting
K.set_image_data_format("channels_first")
print("imgae data format : ",K.image_data_format())

"""
Convolution : (W-F+2P)/S+1
DeConvolution : S*(W-1)+F-P
"""

def g_network_encoder(opt):
    model = Sequential()
    # layer 1
    model.add(ZeroPadding2D(3))
    model.add(Conv2D(opt.nfeature,(7,7),(2,2),"valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 2
    model.add(ZeroPadding2D(2))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2),"valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(2))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def g_network_decoder(opt):
    k = 4 # poke
    model=Sequential()
    # layer 4
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (k,k), (2,2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 5
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (4, 4), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 6
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.n_out, (4, 4), (2, 2), "valid"))
    return model

def phi_network_conv(opt):
    model = Sequential()
    # layer 1
    model.add(ZeroPadding2D(3))
    model.add(Conv2D(opt.nfeature, (7, 7), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 2
    model.add(ZeroPadding2D(2))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(2))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def phi_network_fc(opt):
    model = Sequential()
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(opt.n_latent, activation='tanh'))
    return model

# conditional network
def f_network_encoder(opt):
    model = Sequential()
    # layer 1
    model.add(ZeroPadding2D(3))
    model.add(Conv2D(opt.nfeature, (7, 7), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 2
    model.add(ZeroPadding2D(2))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(2))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model


def f_network_decoder(opt):
    model = Sequential()
    # layer 4
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 5
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (4, 4), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 6
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.n_out, (3, 3), (1, 1), "valid"))
    return model


def encoder_latent(opt):
    model=Sequential()
    model.add(Dense(opt.nfeature))
    return model(input)


# Deterministic Model
class DeterministicModel:
    def __init__(self, opt):
        self.opt = opt
        self.g_network_encoder = g_network_encoder(self.opt)
        self.g_network_decoder = g_network_decoder(self.opt)

    def build(self):
        inputs = Input((self.opt.nc, self.opt.height, self.opt.width))
        outputs = self.g_network_decoder(self.g_network_encoder(inputs))
        model = Model(inputs, outputs)
        return model

    def get_layer(self):
        return [self.g_network_encoder, self.g_network_decoder]

# Latent Variable Model
class LatentResidualModel3Layer:
    def __init__(self, opt):
        self.opt = opt
        self.g_network_encoder = g_network_encoder(self.opt)
        self.g_network_decoder = g_network_decoder(self.opt)
        self.f_network_encoder = f_network_encoder(self.opt)
        self.f_network_decoder = f_network_decoder(self.opt)
        self.phi_network_conv = phi_network_conv(self.opt)
        self.phi_network_fc = phi_network_fc(self.opt)
        self.encoder_latent = encoder_latent(self.opt)

    def build(self):
        inputs = Input(( self.opt.batch_size, self.opt.nc, self.opt.height, self.opt.width))
        inputs = inputs[0]
        targets = inputs[1]
        pred_g = self.g_network_decoder(self.g_network_encoder(inputs))
        model_g = Model(inputs, pred_g)
        r = targets - pred_g
        z = self.phi_network_fc(K.reshape(self.phi_network_conv(r),(self.opt.batch_size, -1)))
        z = K.reshape(z,(self.opt.batch_size, self.opt.n_latent))
        z_emb = self.encoder_latent(z)
        z_emb = K.reshape(z_emb, (self.opt.batch_size, self.opt.nfeature, 1, 1))
        s = self.f_network_encoder(inputs)
        h = s + z_emb
        pred_f = self.f_network_decoder(h)
        model_f = Model(inputs, pred_f)
        return model_g, model_f, z

    def decode(self, inputs, z):
        inputs = K.reshape(inputs, (self.opt.batch_size,
                                    self.opt.ncond * self.opt.nc,
                                    self.opt.height,
                                    self.opt.width))
        z_emb = self.encoder_latent(z)
        z_emb = K.reshape(z_emb, (self.opt.batch_size, self.opt.nfeature))
        s = self.f_network_encoder(inputs)
        h = s + z_emb
        pred = self.f_network_decoder(h)
        return pred







