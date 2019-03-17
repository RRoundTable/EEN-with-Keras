from __future__ import division
import argparse, pdb, os, numpy, imp
from datetime import datetime
import model, utils
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error


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
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
opt = parser.parse_args()


# load data and get dataset-specific parameters
data_config = utils.read_config('config.json').get(opt.task)
data_config['batchsize'] = opt.batch_size
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])

opt.ncond = data_config['ncond']
opt.npred = data_config['npred']
opt.height = data_config['height']
opt.width = data_config['width']
opt.nc = data_config['nc']
opt.phi_fc_size = data_config['phi_fc_size']
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
dataloader = ImageLoader(data_config)


# Set filename based on parameters
opt.save_dir = '{}{}'.format(opt.save_dir, opt.task)
opt.n_in = opt.ncond * opt.nc
opt.n_out = opt.npred * opt.nc
opt.model_filename = '{}/model={}-loss={}-ncond={}-npred={}-nf={}-nz={}-lrt={}'.format(
                    opt.save_dir, opt.model, opt.loss, opt.ncond, opt.npred, opt.nfeature, opt.n_latent, opt.lrt)
print("Saving to " + opt.model_filename)

cond, target, action = dataloader.get_batch("train")

# data
print("cond shape : {}".format(cond.shape)) #  torch.Size([64, 3, 240, 240])
print("target shape:  {}".format(target.shape)) # torch.Size([64, 3, 240, 240])
print("action shape : {}".format(action.shape)) #  torch.Size([64, 5])

# ############################
# ########## train ###########

def train_epoch(model_g,model_f,nsteps):
    total_loss_f, total_loss_g=0,0
    for iter in range(nsteps):
        cond, target, action = dataloader.get_batch("train")
        total_loss_g +=model_g.train(cond,target)
        total_loss_f +=model_f.train(cond, target)
    return total_loss_g/nsteps, total_loss_f/nsteps

def test_epoch(model_g, model_f, nsteps):
    total_loss_f, total_loss_g=0,0
    for iter in range(nsteps):
        cond, target, action = dataloader.get_batch("valid")
        pred_g=model_g(cond)
        pred_f=model_f(cond,target)
        total_loss_g+=model_g.loss_function(target,pred_g)
        total_loss_f+=model_f.loss_function(target,pred_f)
    return total_loss_g/nsteps, total_loss_f/nsteps


def train(model_g,model_f,n_epochs):

    # training
    train_f_loss=[]
    train_g_loss=[]
    # valiation
    val_f_loss=[]
    val_g_loss=[]

    for i in range(0,n_epochs):
        train_g_loss_epoch, train_f_loss_epcoh=train_epoch(model_g,model_f,opt.epoch_size)
        val_g_loss_epoch, val_f_loss_epoch=test_epoch(model_g, model_f, int(opt.epoch_size/5))

        train_g_loss.append(train_g_loss_epoch)
        train_f_loss.append(train_f_loss_epcoh)
        val_f_loss.append(val_f_loss_epoch)
        val_g_loss.append(val_g_loss_epoch)
        print("epoch {} training :: g_loss : {}, f_loss : {}".format(i,
                                                                     train_g_loss_epoch,
                                                                     train_f_loss_epcoh))

if __name__=="__main__":

    model_g=model.DeterministicModel(opt)
    model_f=model.LatentResidualModel3Layer(model_g,opt)
    train(model_g, model_f,500)




