#%%
from matplotlib.pyplot import axis
import tensorflow as tf
#%%
def ELBO_criterion(prob, xhat, x, mean, logvar, prior_means, sigma, num_classes, args):
    # reconstruction error
    if args['bce']:
        error = tf.reduce_sum(- tf.reduce_sum(x * tf.math.log(tf.clip_by_value(xhat, 1e-10, 1.)) + 
                                            (1. - x) * tf.math.log(1. - tf.clip_by_value(xhat, 1e-10, 1.)), axis=[1, 2, 3]))
    else:
        error = tf.reduce_sum(tf.reduce_sum(tf.math.square(x - xhat), axis=[1, 2, 3]) / 2.)
    
    # KL divergence by closed form
    kl1 = tf.reduce_sum(tf.reduce_sum(prob * (tf.math.log(tf.clip_by_value(prob, 1e-10, 1.)) + 
                                              tf.math.log(tf.cast(num_classes, tf.float32))), axis=1))
    kl2 = tf.reduce_sum(tf.reduce_sum(tf.multiply(prob, 
                                                tf.reduce_sum(0.5 * (tf.math.pow(mean - prior_means, 2) / sigma
                                                                    - 1
                                                                    + tf.math.log(sigma)
                                                                    + tf.math.exp(logvar) / sigma
                                                                    - logvar), axis=-1)), axis=-1))
    # kl2 = tf.reduce_sum(tf.reduce_sum(tf.multiply(prob, 
    #                                             0.5 * (tf.reduce_sum(tf.math.pow(mean - prior_means, 2) / sigma, axis=-1)
    #                                                     - args['latent_dim']
    #                                                     + tf.reduce_sum(tf.math.log(sigma))
    #                                                     + tf.reduce_sum(tf.math.exp(logvar) / sigma, axis=-1)
    #                                                     - tf.reduce_sum(logvar, axis=-1))), axis=-1))
    return error, kl1, kl2
#%%