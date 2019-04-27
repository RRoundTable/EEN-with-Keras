## Introduction

This is an unofficial implementation of Error Encoding Networks which is originally developed by Facebook AI Research using Keras.

To alleviate the necessity of GPU resources, the smaller network is used by reducing the size of an input image.

## Thanks to

This project is independently sponsored by [EpiSys Science](http://episci-inc.com/) which research is mainly focused on **uncertainty detection** in deep learning.

## Error Encoding Network

This [paper](https://arxiv.org/pdf/1711.04994.pdf) is branched on a simple idea which is disentangled components of the predictable future state.

As a result, it is able to consistently generate diverse predictions without minimizing the alternating latent space or adversarial training.

## Model structure

The model is trained to alternate minimizing latent variable model.

![structure](./img/een-crop.png)

## Latent variable

![pca](./img/z_pca_dist.png)


## Result

As a result, each decoding image can be represented with a combination of the input image and latent variable which is indicated by red points.

The distance of different latent variables measures the similarity of decoding images.
 
![demo](./results/cond_0.gif)

![demo](./results/cond_11.gif)

## Dependency

- tensorflow-gpu 1.3
- numpy
- matplotlib

## Usage

- train
```python
python train.py [option]
```
- visualize
```python
python visualize.py [option]
```

## Dataset

[Poke](http://ashvin.me/pokebot-website/)

## Reference
- paper : https://arxiv.org/pdf/1711.04994.pdf
- github : https://github.com/mbhenaff/EEN



