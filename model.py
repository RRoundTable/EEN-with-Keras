#-*- coding: utf-8 -*-
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU,\
    Conv2DTranspose,Dense, ZeroPadding2D,Reshape,concatenate,Lambda, Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.python.keras import backend as K
import argparse
import tensorflow as tf
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke', help='breakout | seaquest | flappy | poke | driving')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-model', type=str, default='latent-3layer')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps')
parser.add_argument('-n_latent', type=int, default=4, help='dimensionality of z')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parser.add_argument('-gpu', type=int, default=1)
parser.add_argument('-datapath', type=str, default='.', help='data folder')
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
parser.add_argument('-height', type=int, default=50)
parser.add_argument('-width', type=int, default=50)
parser.add_argument('-nc', type=int, default=3)
parser.add_argument('-npred', type=int, default=1)
parser.add_argument('-n_out', type=int, default=3)
opt = parser.parse_args()

# setting
K.set_image_data_format("channels_first")
print("imgae data format : ",K.image_data_format())

"""
Convolution : (W-F+2P)/S+1
DeConvolution : S*(W-1)+F-P
"""

def g_network_encoder(opt):
    model = Sequential(name="g_encoder")
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
    model=Sequential(name="g_decoder")
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
    model.add(Conv2DTranspose(opt.n_out, (3, 3), (1, 1), "valid"))
    return model

def phi_network_conv(opt):
    model = Sequential(name="phi_conv")
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
    model = Sequential(name="phi_fc")
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
    model = Sequential(name="f_encoder")
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
    model = Sequential(name="f_decoder")
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
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.n_out, (3, 3), (1, 1), "valid"))
    return model


def encoder_latent(opt):
    model=Sequential(name="encoder_latent")
    model.add(Dense(opt.nfeature))
    return model


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
        inputs_ = Input((self.opt.nc, self.opt.height, self.opt.width))
        targets_ = Input((self.opt.nc, self.opt.height, self.opt.width))
        h = Lambda(self.custom_function)([inputs_,targets_])
        pred_f = self.f_network_decoder(h)
        model_f = Model([inputs_,targets_], pred_f)
        return model_f

    def custom_function(self, x):
        inputs = x[0]
        targets = x[1]
        pred_g = self.g_network_decoder(self.g_network_encoder(inputs))
        model_g = Model(inputs, pred_g)
        # residual
        r = Lambda((lambda x: x[1] - model_g(x[0])))([inputs, targets])
        out_dim = K.int_shape(self.phi_network_conv(r))  # shape=(?, 64, 7, 7)
        z = self.phi_network_fc(K.reshape(self.phi_network_conv(r),
                                          (self.opt.batch_size, out_dim[1] * out_dim[2] * out_dim[3])))
        z = K.reshape(z, (self.opt.batch_size, self.opt.n_latent))
        z_emb = self.encoder_latent(z)
        z_emb = K.reshape(z_emb, (self.opt.batch_size, self.opt.nfeature, 1, 1))
        s = self.f_network_encoder(inputs)
        return Lambda((lambda x: x[0] + x[1]))([s, z_emb])


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


if __name__ == '__main__':
    EEN = LatentResidualModel3Layer(opt)
    model = EEN.build()
    model.compile()





