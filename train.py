from __future__ import division
import argparse, pdb, os, numpy, imp
from datetime import datetime
import model, utils
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
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
parser.add_argument('-log_path_f', type=str, default='./logs_f', help='tensorboard : conditional network')
parser.add_argument('-log_path_g', type=str, default='./logs_g', help='tensorboard : deterministic network')
opt = parser.parse_args()

# load data and get dataset-specific parameters
data_config = utils.read_config('config.json').get(opt.task)
data_config['batchsize'] = opt.batch_size
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])

opt.ncond = data_config['ncond']
opt.npred = data_config['npred']
# opt.height = data_config['height']
# opt.width = data_config['width']
opt.height=50
opt.width=50
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

cond, target, action = dataloader.get_batch("train")

# data shape
print("cond shape : {}".format(cond.shape)) #  torch.Size([64, 3, 240, 240])
print("target shape:  {}".format(target.shape)) # torch.Size([64, 3, 240, 240])
print("action shape : {}".format(action.shape)) #  torch.Size([64, 5])

if not os.path.exists(opt.log_path_f):
    os.mkdir(opt.log_path_f)
if not os.path.exists(opt.log_path_g):
    os.mkdir(opt.log_path_g)

if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)

callback_f=TensorBoard(opt.log_path_f)
callback_g=TensorBoard(opt.log_path_g)


def named_logs(names, logs):
  result = {}
  for l in zip(names, logs):
    result[l[0]] = l[1]
  return result

def write_log(callback, names, logs, batch_no):
    for name,value in zip(names,logs):
        summary=tf.Summary()
        summary_value=summary.value.add()
        summary_value.simple_value=value
        summary_value.tag=name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def train_epoch(model_f, mode, nsteps):
    total_loss_f  =  0
    if mode == 'latent-3layer' :
        for iter in range(nsteps):
            cond, target, action = dataloader.get_batch("train")
            total_loss_f += model_f.train_on_batch([cond, target], target) # conditional network
    else:
        for iter in range(nsteps):
            cond, target, action = dataloader.get_batch("train")
            total_loss_f += model_f.train_on_batch(cond, target) # base network
    return total_loss_f/nsteps

def test_epoch(model_f, mode, nsteps):
    total_loss_f = 0
    if mode == 'latent-3layer' :
        for iter in range(nsteps):
            cond, target, action = dataloader.get_batch("train")
            total_loss_f += model_f.test_on_batch([cond, target], target) # conditional network
    else:
        for iter in range(nsteps):
            cond, target, action = dataloader.get_batch("train")
            total_loss_f += model_f.test_on_batch(cond, target) # base network
    return total_loss_f/nsteps


def train(model_f, callback_f, mode, n_epochs):
    # mode
    if mode == 'latent-3layer':
        save_path = opt.model_filename_f
    else:
        save_path = opt.model_filename_g

    # training
    train_f_loss=[]
    # valiation
    best_valid_loss_f = 0.2
    val_f_loss=[]
    names=['train_loss', "val_loss"]
    for i in range(0,n_epochs):
        train_f_loss_epoch=train_epoch(model_f, mode, opt.epoch_size)
        val_f_loss_epoch=test_epoch(model_f, mode, int(opt.epoch_size/5))
        train_f_loss.append(train_f_loss_epoch)
        val_f_loss.append(val_f_loss_epoch)
        if val_f_loss_epoch<best_valid_loss_f :
            best_valid_loss_f=val_f_loss_epoch
            model_f.save_weights(save_path, overwrite=True)
        if i%5==0:
            model_f.save_weights(save_path, overwrite=True)
        # write log : tensorboard
        print("Write summary...")
        callback_f.on_epoch_end(i,named_logs(names, [train_f_loss_epoch,val_f_loss_epoch]))
        print("epoch {} training :: f_loss : {}".format(i, train_f_loss_epoch))
        print("epoch {} validation ::  f_loss : {}".format(i, val_f_loss_epoch))

if __name__=="__main__":
    een = model.LatentResidualModel3Layer(opt)
    base = model.BaselineModel3Layer(opt)

    model_f = een.build()
    base_model = base.build()

    if opt.loss == 'l1':
        loss = mean_absolute_error
    elif opt.loss == 'l2':
        loss = mean_squared_error

    # load trained weight
    if os.path.exists(opt.model_filename_f):
        # model_g.trainable=False
        print("load weight model_f ....")
        model_f.load_weights(opt.model_filename_f)

    optimizer = Adam(opt.lrt)
    model_f.compile(optimizer=optimizer, loss= loss)
    base_model.compile(optimizer=optimizer, loss = loss)
    callback_f.set_model(model_f)
    callback_g.set_model(base_model)
    # our model
    if opt.model == 'latent-3layer':
        print("{} model train!....".format(opt.model))
        train(model_f, callback_f, opt.model,500)
    else:
        print("{} model train!....".format(opt.model))
        train(base_model, callback_g, opt.model, 500)



