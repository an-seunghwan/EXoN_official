#%%
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

import random
import scipy
#%%
def crop_center(img, cropx, cropy):
    if len(img.shape) == 2:
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]
    elif len(img.shape) == 3:
        y, x, b = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx, :]

def crop_random(img, cropx, cropy):
    if len(img.shape) == 2:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1 : x1 + cropx, y1 : y1 + cropy]
    elif len(img.shape) == 3:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1 : x1 + cropx, y1 : y1 + cropy, :]

def augment(image):
    npimage = image.numpy()
    rotation = [random.randrange(-25, 25) for i in range(len(npimage))]
    rotatedImg = [scipy.ndimage.interpolation.rotate(im, rotation[i], axes=(0, 1), mode='nearest') for i, im in enumerate(npimage)]
    rotatedImgCentered = [crop_center(im, 28, 28) for im in rotatedImg]
    paddedImg = [np.pad(im, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=-1) for im in rotatedImgCentered]
    cropped = np.array([crop_random(im, 28, 28) for im in paddedImg])
    return tf.stack(cropped, axis=0)
#%%
def non_smooth_mixup(image, label, mix_weight):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    label_shuffle = tf.gather(label, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    
    return image_mix, label_shuffle
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
#%%