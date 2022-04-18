#%%
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf

#%%
@tf.function
def augment(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, paddings=[(0, 0), (2, 2), (2, 2), (0, 0)], mode="REFLECT")
    # x = tf.image.random_saturation(x, lower=0.6, upper=1.4)
    x = tf.map_fn(
        lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)),
        x,
        parallel_iterations=cpu_count(),
    )
    return x


#%%
def non_smooth_mixup(image, label, mix_weight):
    shuffled_indices = tf.random.shuffle(
        tf.range(start=0, limit=tf.shape(image)[0], dtype=tf.int32)
    )

    image_shuffle = tf.gather(image, shuffled_indices)
    label_shuffle = tf.gather(label, shuffled_indices)

    image_mix = mix_weight * image_shuffle + (1.0 - mix_weight) * image

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
