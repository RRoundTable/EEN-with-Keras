# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Apr. 2019 by wontak ryu.
ryu071511@gmail.com.
https://github.com/RRoundTable/EEN-with-Keras.

Make gif file.
'''

import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import imageio
from scipy import misc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=64)
opt = parser.parse_args()

def img_file_to_gif(img_file_lst, output_file_name):
    imgs_array = [np.array(imageio.imread(img_file)) for img_file in img_file_lst]
    imageio.mimsave(output_file_name, imgs_array, duration=0.5)


if __name__ == "__main__":
    result_file = []
    for i in range(opt.batch_size):
        result_file.append(glob.glob(('./results/poke/latent/ep/ep*_{}.png'.format(i))))

    for i in range(opt.batch_size):
        tpath = './results/poke/cond_{}.gif'.format(i)
        img_file_to_gif(result_file[i], tpath)

