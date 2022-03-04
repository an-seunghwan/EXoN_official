#%%
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
#%%
@tf.function
def augment(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, paddings=[(0, 0),
                            (2, 2),
                            (2, 2), 
                            (0, 0)], mode='REFLECT')
    # x = tf.image.random_saturation(x, lower=0.6, upper=1.4)
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    return x
#%%
def non_smooth_mixup(image, label, mix_weight):
    shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
    image_shuffle = tf.gather(image, shuffled_indices)
    label_shuffle = tf.gather(label, shuffled_indices)
    
    image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
    
    return image_mix, label_shuffle
#%%
# def label_smoothing(image, label, mix_weight):
#     shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
#     image_shuffle = tf.gather(image, shuffled_indices)
#     label_shuffle = tf.gather(label, shuffled_indices)
    
#     image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
#     label_mix = mix_weight * label_shuffle + (1. - mix_weight) * label
    
#     return image_mix, label_mix, label_shuffle
# #%%
# def non_smooth_mixup(image, prob, mix_weight):
#     shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32))
    
#     image_shuffle = tf.gather(image, shuffled_indices)
#     prob_shuffle = tf.gather(prob, shuffled_indices)
    
#     image_mix = mix_weight * image_shuffle + (1. - mix_weight) * image
#     prob_mix = mix_weight * prob_shuffle + (1. - mix_weight) * prob
    
#     return image_mix, prob_mix
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