#%%
import tensorflow as tf
#%%
def ELBO_criterion(prob, xhat, x, mean, logvar, beta, prior_means, sigma_vector, num_classes, args):
    # reconstruction error
    error = tf.reduce_mean(tf.reduce_sum(tf.math.square(x - xhat), axis=[1, 2, 3]) / (2 * beta), axis=-1)
    
    # KL divergence by closed form
    kl = tf.reduce_mean(tf.reduce_sum(prob * tf.math.log(prob * num_classes + 1e-8), axis=1))
    kl += tf.reduce_mean(tf.reduce_sum(tf.multiply(prob, 
                                                0.5 * (tf.reduce_sum(tf.math.pow(mean - prior_means, 2) / sigma_vector, axis=-1)
                                                        - args['latent_dim']
                                                        + tf.reduce_sum(tf.math.log(sigma_vector))
                                                        + tf.reduce_sum(tf.math.exp(logvar) / sigma_vector, axis=-1)
                                                        - tf.reduce_sum(logvar, axis=-1))), axis=-1))
    return error, kl
#%%