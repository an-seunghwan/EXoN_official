#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class FeatureExtractor(K.models.Model):
    def __init__(self, name="FeatureExtractor", **kwargs):
        super(FeatureExtractor, self).__init__(name=name, **kwargs)
        self.net = K.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same'), # 16x16
                layers.ReLU(),
                layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'), # 8x8
                layers.ReLU(),
                layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'), # 4x4
                layers.ReLU(),
                layers.Flatten(),
                layers.Dense(256, activation='linear'),
                layers.ReLU(),
            ]
        )
        
    @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        return h
#%%
class Decoder(K.models.Model):
    def __init__(self, activation='tanh', name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.net = K.Sequential(
            [
                layers.Dense(256, activation='linear'),
                layers.ReLU(),
                layers.Dense(64*4*4, activation='linear'),
                layers.ReLU(),
                layers.Reshape((4, 4, 64)),
                
                layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same'),
                layers.ReLU(),
                layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same'),
                layers.ReLU(),
                layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation=activation),
            ]
        )
    
    @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        return h
#%%
class MixtureVAE(K.models.Model):
    def __init__(self, 
                 args,
                 num_classes=10,
                 latent_dim=2, 
                 activation='tanh',
                 input_dim=(None, 32, 32, 1), 
                 hard=True,
                 name='MixtureVAE', **kwargs):
        super(MixtureVAE, self).__init__(name=name, **kwargs)
        self.hard = hard
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.feature_extractor = FeatureExtractor()
        self.mean_layer = [layers.Dense(latent_dim, activation='linear') for _ in range(num_classes)]
        self.logvar_layer = [layers.Dense(latent_dim, activation='linear') for _ in range(num_classes)]
        self.classifier = layers.Dense(num_classes, activation='softmax')
        self.decoder = Decoder(activation)
    
    def sample_gumbel(self, shape): 
        U = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(U + 1e-8) + 1e-8)

    def gumbel_max_sample(self, probs): 
        y = tf.math.log(probs + 1e-8) + self.sample_gumbel(tf.shape(probs))
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 1, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y
    
    def encode(self, x, training=True):
        h = self.feature_extractor(x, training=training)
        mean = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.mean_layer])
        logvar = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.logvar_layer])
        prob = self.classifier(h, training=training)
        epsilon = tf.random.normal((tf.shape(x)[0], self.num_classes, self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        y = self.gumbel_max_sample(prob)
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        return mean, logvar, prob, y, z, z_tilde
    
    def classify(self, x, training=True):
        h = self.feature_extractor(x, training=training)
        prob = self.classifier(h, training=training)
        return prob
    
    def decode(self, z, training=True):
        xhat = self.decoder(z, training=training) 
        return xhat
        
    @tf.function
    def call(self, x, training=True):
        h = self.feature_extractor(x, training=training)
        mean = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.mean_layer])
        logvar = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.logvar_layer])
        prob = self.classifier(h, training=training)
        
        epsilon = tf.random.normal((tf.shape(x)[0], self.num_classes, self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        # assert z.shape == (tf.shape(x)[0], self.num_classes, self.latent_dim)
        
        y = self.gumbel_max_sample(prob)
        # assert y.shape == (tf.shape(x)[0], self.num_classes)
        
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        # assert z_tilde.shape == (tf.shape(x)[0], self.latent_dim)
        
        xhat = self.decoder(z_tilde, training=training) 
        # assert xhat.shape == (tf.shape(x)[0], self.input_dim[1], self.input_dim[2], self.input_dim[3])
        
        return mean, logvar, prob, y, z, z_tilde, xhat
#%%