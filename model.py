#-*- coding: utf-8 -*-
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU,\
    Conv2DTranspose,Dense, ZeroPadding2D,Reshape,Input
from tensorflow.python.keras.activations import tanh
from tensorflow.python.keras.models import Sequential, Model


def g_network_encoder(input,opt):
    g_network_encoder = Sequential(
        # layer1
        ZeroPadding2D((3,3)),
        Conv2D(opt.nfeature,(7,7),(2,2),"valid"), # padding=3
        BatchNormalization(),
        ReLU(),
        # layer2
        ZeroPadding2D((2, 2)),
        Conv2D(opt.nfeature, (5, 5), (2, 2),"valid"),
        BatchNormalization(),
        ReLU(),
        # layer3
        ZeroPadding2D((2, 2)),
        Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"),
        BatchNormalization(),
        ReLU()
    )
    return g_network_encoder(input)

def g_network_decoder(input,opt):
    g_network_decoder = Sequential(
        # layer 4
        ZeroPadding2D((3, 3)),
        Conv2DTranspose(opt.nfeature,(7,7),(2,2),"valid"),
        BatchNormalization(),
        ReLU(),
        # layer 5
        ZeroPadding2D((2, 2)),
        Conv2DTranspose(opt.nfeature, (5, 5), (2, 2),"valid"),
        BatchNormalization(),
        ReLU(),
        # layer 6
        ZeroPadding2D((2, 2)),
        Conv2DTranspose(opt.nfeature, (5, 5), (2, 2), "valid"),
        BatchNormalization(),
        ReLU()
    )
    return g_network_decoder(input)


def phi_network_conv(input,opt):
    phi_network_conv = Sequential(
        ZeroPadding2D((3, 3)),
        Conv2D(opt.nfeature,(7,7),(2,2),"valid"),
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D((2, 2)),
        Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"),
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D((2, 2)),
        Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"),
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D((2, 2)),
        Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"),
        BatchNormalization(),
        ReLU()
    )
    return phi_network_conv(input)

def phi_network_fc(input,opt):
    phi_network_fc = Sequential(
        Dense(1000),
        BatchNormalization(),
        ReLU(),
        Dense(1000),
        BatchNormalization(),
        ReLU(),
        Dense(opt.n_latent),
        tanh()
    )
    return phi_network_fc(input)

# conditional network
def f_network_encoder(input,opt):
    f_network_encoder = Sequential(
        ZeroPadding2D((3,3)),
        Conv2D(opt.nfeature,(7,7),(2,2),"valid"),
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D((2, 2)),
        Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"),
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D((2, 2)),
        Conv2D(opt.nfeature, (5, 5), (2, 2), "valid"),
        BatchNormalization(),
        ReLU(),
    )
    return f_network_encoder(input)

def f_network_decoder(input,opt,k):
    f_network_decoder = Sequential(
        ZeroPadding2D((1,1)),
        Conv2DTranspose(opt.nfeature,(k,k),(2,2),"valid"),
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D((1, 1)),
        Conv2DTranspose(opt.nfeature, (4, 4), (2, 2), "valid"),
        BatchNormalization(),
        ReLU(),
        ZeroPadding2D((1, 1)),
        Conv2DTranspose(opt.nfeature, (4, 4), (2, 2), "valid")
    )
    return f_network_decoder(input)

def encoder_latent(input, opt):
    Linear=Sequential(
        Dense(opt.nfeature)
    )
    return Linear(input)


# EEN_latent variable
def LatentResidualModel3Layer(input, target, opt):
    input=Reshape((opt.ncond*opt.nc,opt.height, opt.width))(input) # batch_size, shape
    target=Reshape((opt.npred*opt.nc,opt.height,opt.width))(target)
    # encoder+ decode
    g_pred=g_network_decoder(g_network_encoder(input,opt))

    # variable : don't pass gradient to g from phi
    g_pred_v=g_pred # gradient가 전달되지 않게 하기 :

    # residual
    r=target-Input(g_pred_v)

    # dimension!
    z=phi_network_fc(phi_network_conv(r,opt),opt)
    z=Reshape(opt.n_latent)(z)
    z_emb=Reshape(opt.nfeature)(encoder_latent(z,opt))

    s=f_network_encoder(input,opt)
    h=s+Reshape(opt.nfeature,1,1)(z_emb)

    k = 4
    if opt.task == 'breakout' or opt.task == 'seaquest':
        # need this for output to be the right size
        k = 3

    pred_f=f_network_decoder(h,opt,k)

    # return Model
    f_model=Model(input,pred_f, name="f_network")
    g_model=Model(input, g_pred, name="g_network")


    return f_model, g_model, z



