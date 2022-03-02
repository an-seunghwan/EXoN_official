#%%
import tensorflow as tf
#%%
def ELBO_criterion(prob, xhat, x, mean, logvar, prior_means, sigma_vector, num_classes, args):
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
                                                tf.reduce_sum(0.5 * (tf.math.pow(mean - prior_means, 2) / sigma_vector
                                                                    - 1
                                                                    + tf.math.log(sigma_vector)
                                                                    + tf.math.exp(logvar) / sigma_vector
                                                                    - logvar), axis=-1)), axis=-1))
    return error, kl1, kl2
#%%