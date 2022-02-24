#%%
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

import random
import scipy
#%%
# @tf.function
# def augment(x):
#     x = tf.image.random_flip_left_right(x)
#     x = tf.pad(x, paddings=[(0, 0),
#                             (4, 4),
#                             (4, 4), 
#                             (0, 0)], mode='REFLECT')
#     x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
#     return x
#%%
# Crops the center of the image
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

# Take a random crop of the image
def crop_random(img, cropx, cropy):
    # takes numpy input
    if len(img.shape) == 2:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1 : x1 + cropx, y1 : y1 + cropy]
    elif len(img.shape) == 3:
        x1 = random.randint(0, img.shape[0] - cropx)
        y1 = random.randint(0, img.shape[1] - cropy)
        return img[x1 : x1 + cropx, y1 : y1 + cropy, :]

def augment(image):
    npimage = image.numpy() # input is tensor
    rotation = [random.randrange(-25, 25) for i in range(len(npimage))]
    rotatedImg = [scipy.ndimage.interpolation.rotate(im, rotation[i], axes=(0, 1), mode='nearest') for i, im in enumerate(npimage)]
    # crop image to 28x28 as rotation increases size
    rotatedImgCentered = [crop_center(im, 28, 28) for im in rotatedImg]
    # pad image by 3 pixels on each edge (-0.42421296 background color)
    paddedImg = [np.pad(im, ((4, 4), (4, 4), (0, 0)), 'constant', constant_values=-1) for im in rotatedImgCentered]
    # randomly crop from padded image
    cropped = np.array([crop_random(im, 28, 28) for im in paddedImg])
    return tf.stack(cropped, axis=0)
#%%
def label_smoothing(image, label, mix_weight):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    label_shuffle = tf.gather(label, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    label_mix = mix_weight * label_shuffle + (1. - mix_weight) * label
    
    return image_mix, label_mix, label_shuffle
#%%
def non_smooth_mixup(image, prob, mix_weight):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    prob_shuffle = tf.gather(prob, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    prob_mix = mix_weight * prob_shuffle + (1. - mix_weight) * prob
    
    return image_mix, prob_mix
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
        
# def weight_decay(model, decay_rate):
#     for var in model.trainable_variables:
#         var.assign(var * (1. - decay_rate))
#%%