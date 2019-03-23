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
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2),"valid"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # layer 3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (5, 5), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def g_network_decoder(opt):
    model=Sequential()
    # layer 4
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature,(5,5),(2,2),"valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 5
    #model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (4, 4), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 6
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.n_out, (3, 3), (1, 1), "valid"))
    return model

def phi_network_conv(opt):
    model = Sequential()
    # layer 1
    model.add(ZeroPadding2D(3))
    model.add(Conv2D(opt.nfeature, (7, 7), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 2
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (5, 5), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

# def phi_network_conv(input,opt):
#     model = Sequential()
#     # layer 1
#     model.add(ZeroPadding2D(3))
#     model.add(Conv2D(opt.nfeature, (7, 7), (2, 2), "valid"))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     # layer 2
#     model.add(ZeroPadding2D(2))
#     model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     # layer 3
#     model.add(ZeroPadding2D(2))
#     model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     # layer 4
#     model.add(ZeroPadding2D(2))
#     model.add(Conv2D(opt.nfeature, (6, 6), (2, 2), "valid"))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     return model(input)

def phi_network_decoder(opt):
    model = Sequential()
    # layer 4
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (3, 3), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 5
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (2, 2), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 6
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (2, 2), (2, 2), "valid"))
    return model



def phi_network_fc(opt):
    model = Sequential()
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(opt.n_latent, activation=tanh))
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
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (5, 5), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model


def f_network_decoder(opt):
    model = Sequential()
    # layer 4
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (5, 5), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 5
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (4, 4), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 6
    # model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.n_out, (3, 3), (1, 1), "valid"))
    return model


def encoder_latent(input, opt):
    model=Sequential()
    model.add(Dense(opt.nfeature))

    return model(input)


class LatentResidualModel3Layer(Model):
    def __init__(self,opt, deterministic):
        super(LatentResidualModel3Layer, self).__init__()
        self.opt=opt
        self.deterministic=deterministic # 학습시키지 않을 layer
        self.deterministic.trainable=False
        self.phi_network_conv=phi_network_conv(opt)
        # self.phi_network_fc=phi_network_fc
        # self.phi_network_decoder=phi_network_decoder
        # self.encoder_latent=encoder_latent
        # self.f_network_encoder=f_network_encoder
        self.f_network_decoder=f_network_decoder(opt)

        self.optimizer=Adam()


        # self.g_network_encoder=g_network_encoder


    def call(self,x):
        """ Forward """
        # [x,y]=Lambda(lambda x:[x[0],x[1]], output_shape=[-1,self.opt.ncond,self.opt.nc,self.opt.height,self.opt.width])(x)
        inputs = Reshape((self.opt.ncond * self.opt.nc, self.opt.height, self.opt.width))(x)  # batch_size, shape
        target = Reshape((self.opt.npred * self.opt.nc, self.opt.height, self.opt.width))(self.y)
        g_pred_v = K.variable(self.deterministic(inputs), name="g_pred_v")
        r=K.abs(g_pred_v-target) # residual
        z=self.phi_network_conv(r)
        s = self.deterministic.get_layer()[0](inputs)
        h=concatenate([s,z], axis=1) # concatenate
        # shape=K.int_shape(s)
        # h=Reshape((shape[1]*2, shape[2],shape[3]))(h)
        pred_f = self.f_network_decoder(h)
        return pred_f

    def get_config(self):
        config=super(LatentResidualModel3Layer,self).get_config()
        config['input_shape']=[self.opt.batch_size, self.opt.nc, self.opt.height, self.opt.width]
        return config

    # 접근 불가능
    def set_input_shape(self):
        self.input_shape=[self.opt.batch_size, self.opt.nc, self.opt.height, self.opt.width]


    def loss_function(self,pred,true):
        """return loss(tensor)"""

        pred = K.flatten(pred)
        y = K.flatten(true)
        if self.opt.loss == 'l1':
            loss = mean_absolute_error(y, pred)
        elif self.opt.loss == 'l2':
            loss = mean_squared_error(y, pred)
        return loss

    def update(self, x, y):
        """ train and return loss(scalar)"""
        loss = self.loss_function(x, y)
        updates=self.optimizer.get_updates(loss,
                                   self.phi_network_conv.trainable_variables +
                                   self.f_network_decoder.trainable_variables)
        return updates

    def get_target(self,y):
        # input layer
        self.y=y

    def get_input(self,x):
        self.x=x


class DeterministicModel(Model):

    def __init__(self, opt):
        super(DeterministicModel, self).__init__()
        self.opt = opt
        self.g_network_encoder=g_network_encoder(opt)
        self.g_network_decoder=g_network_decoder(opt)
        self.optimizer=Adam()
        # self.updates=self.g_network_encoder.trainable_variables+self.g_network_decoder
        # self.loss=0
        # self.gradient_descnt=K.function([self.loss],[],updates=self.updates)

    def call(self,x):
        """ Forward """
        inputs = Reshape((self.opt.ncond * self.opt.nc, self.opt.height, self.opt.width))(x)  # batch_size, shape
        # encoder+ decode
        k = 4
        if self.opt.task == 'breakout' or self.opt.task == 'seaquest':
            # need this for output to be the right size
            k = 3
        g_pred = self.g_network_decoder(self.g_network_encoder(inputs))
        return g_pred

    def get_layer(self):
        return [self.g_network_encoder, self.g_network_decoder]

    def loss_function(self,pred,true):
        """return loss(tensor)"""
        pred=K.flatten(pred)
        y=K.flatten(true)
        if self.opt.loss == 'l1':
            loss = mean_absolute_error(y, pred)
        elif self.opt.loss == 'l2':
            loss = mean_squared_error(y, pred)
        return loss

    # def train(self,x,y):
    #     """ train and return loss(scalar)"""
    #     self.loss=self.loss_function(x,y)
    #     updates=self.optimizer.get_updates(self.loss,
    #                                self.g_network_encoder.trainable_variables+
    #                                self.g_network_decoder.trainable_variables)
    #
    #     print("update : {}".format(updates))
    #     # self.gradient_descent(self.loss)
    #
    #     loss=K.get_value(self.loss)
    #     return loss







