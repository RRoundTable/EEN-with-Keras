import argparse, pdb, os, numpy, glob, imp, imageio
from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error
from datetime import datetime
import matplotlib as mpi
mpi.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import model, utils
import numpy as np

# Training settings
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
parser.add_argument('-f_model', type=str, default='f')
parser.add_argument('-g_model', type=str, default='g')
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
parser.add_argument('-num', type=int, default=50, help='How much to visualize')
opt = parser.parse_args()

# load data and get dataset-specific parameters
data_config = utils.read_config('config.json').get(opt.task)
data_config['batchsize'] = opt.batch_size
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])

opt.ncond = data_config['ncond']
opt.npred = data_config['npred']
opt.height = 50
opt.width = 50
opt.nc = data_config['nc']
opt.phi_fc_size = data_config['phi_fc_size']
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
dataloader = ImageLoader(data_config)

# Set filename based on parameters
opt.save_dir = '{}{}'.format(opt.save_dir, opt.task)
opt.n_in = opt.ncond * opt.nc
opt.n_out = opt.npred * opt.nc
opt.model_filename = '{}/model={}-loss={}-ncond={}-npred={}-nf={}-nz={}-lrt={}.h5'.format(
                    opt.save_dir, opt.model, opt.loss, opt.ncond, opt.npred, opt.nfeature, opt.n_latent, opt.lrt)
opt.model_filename_f = '{}/model={}-loss={}-ncond={}-npred={}-nf={}-nz={}-lrt={}.h5'.format(
                    opt.save_dir, opt.f_model, opt.loss, opt.ncond, opt.npred, opt.nfeature, opt.n_latent, opt.lrt)
opt.model_filename_g = '{}/model={}-loss={}-ncond={}-npred={}-nf={}-nz={}-lrt={}.h5'.format(
                    opt.save_dir, opt.g_model, opt.loss, opt.ncond, opt.npred, opt.nfeature, opt.n_latent, opt.lrt)
print("Saving to " + opt.model_filename)

def plot_seq(cond, pred, path = None):
    """
    :param cond: input image
    :param pred: predict image
    :return: grid image
    """
    cond_pred = np.concatenate((cond, pred), axis=1)  # channel
    cond_pred = np.reshape(cond_pred, newshape=(-1, opt.nc, opt.height, opt.width))

    # matplotlib : subplot
    total = len(cond_pred)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    i = 0
    for row in ax:
        for col in row:
            col.imshow(cond_pred[i])

    plt.savefig(path) # 위치지정하기
    return fig

if __name__ == "__main__":

    een = model.LatentResidualModel3Layer(opt)
    base = model.BaselineModel3Layer(opt)

    # build model
    model_f = een.build()
    base_model = base.build()

    if opt.loss == 'l1':
        loss = mean_absolute_error
    elif opt.loss == 'l2':
        loss = mean_squared_error

    if opt.model == 'latent-3layer':
        # load weights : latent model
        print("{} model loading ...".format(opt.model))
        model_f.load_weights(opt.model_filename_f)
        een.load_weights(model_f) # transfer weight
        model_z = een.get_model_z() # latent variable model

        # extract some z vectors from the training set
        zlist, alist = [], []
        for _ in range(opt.num):
            cond, target, action = dataloader.get_batch("train")
            # our model : latent model
            z = model_z([cond, target])
            zlist.append(z)
            alist.append(action)
        znp = np.array(zlist)
        # if more 2D, compute PCA so we can visualize the z distribution
        if znp.shape[1] > 2:
            pca = PCA(n_components=2)
            znp = pca.fit(znp).transform(znp)

        plt.scatter(znp[:, 0], znp[:, 1], s=2)
        plt.savefig('{}/z_pca_dist.png'.format(opt.save_dir))
        plt.clf()

        cond, target, action = dataloader.get_batch("test")
        pred_f = model_f([cond, target])
        error = target - pred_f
        for b in range(opt.batch_size):
            truth_path = opt.save_dir + "/latent/truth_{}.png".format(b)
            plot_seq(cond[b], target[b], truth_path)
            pred_path = opt.save_dir + "/latent/pred_{}.png".format(b)
            plot_seq(cond[b], pred_f[b], pred_path)

        # different z vectors
        nz = opt.num
        mov = []
        for idx in range(nz):
            mov.append([])
            print("-------{}------".format(idx))
            pred_z = een.decode(cond, z)
            for b in range(opt.batch_size):
                save_path = opt.save_dir + "/ep{}.png".format(b)
                img = plot_seq(cond[b], pred_z[b], save_path)
                mov[-1].append(img)

        # make movie
        # for idx in range(nz):
        #     mov[idx] =

    else:
        # load weights : base model
        print("{} model loading ...".format(opt.model))
        base_model.load_weights(opt.model_filename_g)
        model_f.compile(optimizer="Adam", loss=loss)
        base_model.compile(optimizer='Adam', loss=loss)

        cond, target, action = dataloader.get_batch("test")
        # base model : base
        pred_base = base_model(cond)
        error_base = target - pred_base
        for b in range(opt.batch_size):
            truth_path = opt.save_dir + "/base/truth_{}.png".format(b)
            plot_seq(cond[b], target[b], truth_path)
            pred_path = opt.save_dir + "/base/pred_{}.png".format(b)
            plot_seq(cond[b], pred_base[b], pred_path)

