#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class Encoder(K.models.Model):
    def __init__(self, latent_dim, num_classes, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.net = K.Sequential(
            [
                layers.Flatten(),
                layers.Dense(256, activation='linear'),
                layers.ReLU(),
                layers.Dense(128, activation='linear'),
                layers.ReLU(),
            ]
        )
        
        self.mean_layer = [layers.Dense(latent_dim, activation='linear') for _ in range(num_classes)]
        self.logvar_layer = [layers.Dense(latent_dim, activation='softplus') for _ in range(num_classes)]
    
    # @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        mean = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.mean_layer])
        logvar = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.logvar_layer])
        return mean, logvar
#%%
class Classifier(K.models.Model):
    def __init__(self, num_classes, name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.stddev = 0.2
        self.nets = K.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same'), 
                layers.BatchNormalization(),
                layers.LeakyReLU(0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'), 
                layers.BatchNormalization(),
                layers.LeakyReLU(0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'), 
                layers.BatchNormalization(),
                layers.LeakyReLU(0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=0.5),
                
                layers.GlobalAveragePooling2D(),
                
                layers.Dense(64, activation='linear'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Dense(num_classes, activation='softmax'),
            ]
        )
    
    # @tf.function
    def call(self, x, noise=False, training=True):
        h = x
        if noise:
            # Gaussian noise on input layer
            epsilon = tf.random.normal(shape=tf.shape(h), stddev=self.stddev)
            h += epsilon
        h = self.nets(h, training=training)
        return h
#%%
class Decoder(K.models.Model):
    def __init__(self, activation='tanh', name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.net = K.Sequential(
            [
                layers.Dense(128, activation='linear'),
                layers.ReLU(),
                layers.Dense(256, activation='linear'),
                layers.ReLU(),
                layers.Dense(784, activation=activation),
                layers.Reshape((28, 28, 1)),
            ]
        )
    
    # @tf.function
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
                 input_dim=(None, 28, 28, 1), 
                 hard=True,
                 name='MixtureVAE', **kwargs):
        super(MixtureVAE, self).__init__(name=name, **kwargs)
        self.hard = hard
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.encoder = Encoder(latent_dim, num_classes)
        self.classifier = Classifier(num_classes)
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
    
    def encode(self, x, noise=False, training=True):
        mean, logvar = self.encoder(x, training=training)
        epsilon = tf.random.normal((tf.shape(x)[0], self.num_classes, self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        prob = self.classifier(x, noise=noise, training=training)
        y = self.gumbel_max_sample(prob)
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        return mean, logvar, prob, y, z, z_tilde
    
    def classify(self, x, noise=False, training=True):
        prob = self.classifier(x, noise=noise, training=training)
        return prob
    
    def decode(self, z, training=True):
        xhat = self.decoder(z, training=training) 
        return xhat
        
    @tf.function
    def call(self, x, noise=False, training=True):
        mean, logvar = self.encoder(x, training=training)
        epsilon = tf.random.normal(shape=(tf.shape(x)[0], self.num_classes, self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        # assert z.shape == (tf.shape(x)[0], self.num_classes, self.latent_dim)
        
        prob = self.classifier(x, noise=noise, training=training)
        y = self.gumbel_max_sample(prob)
        # assert y.shape == (tf.shape(x)[0], self.num_classes)
        
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        # assert z_tilde.shape == (tf.shape(x)[0], self.latent_dim)
        
        xhat = self.decoder(z_tilde, training=training) 
        # assert xhat.shape == (tf.shape(x)[0], self.input_dim[1], self.input_dim[2], self.input_dim[3])
        
        return mean, logvar, prob, y, z, z_tilde, xhat
#%%
# class Classifier(K.models.Model):
#     def __init__(self, num_classes, name="Classifier", **kwargs):
#         super(Classifier, self).__init__(name=name, **kwargs)
#         self.stddev = 0.3
#         self.feature_nets = [
#             K.Sequential(
#                 [
#                     layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same'), 
#                     layers.BatchNormalization(),
#                     layers.LeakyReLU(0.01),
#                 ]
#             ),
#             K.Sequential(
#                 [
#                     layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'), # 14x14
#                     layers.BatchNormalization(),
#                     layers.LeakyReLU(0.01),
#                 ]
#             ),
#             K.Sequential(
#                 [
#                     layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'), 
#                     layers.BatchNormalization(),
#                     layers.LeakyReLU(0.01),
#                 ]
#             ),
#             K.Sequential(
#                 [
#                     layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'), 
#                     layers.BatchNormalization(),
#                     layers.LeakyReLU(0.01),
#                 ]
#             ),    
#             K.Sequential(
#                 [
#                     layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'), # 7x7
#                 ]
#             ),
#             K.Sequential(
#                 [
#                     layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'), 
#                     layers.BatchNormalization(),
#                     layers.LeakyReLU(0.01),
#                 ]
#             ),    
#         ]
#         self.nets = K.Sequential(
#             [
#                 layers.Flatten(),
#                 layers.Dense(128, activation='linear'),
#                 layers.BatchNormalization(),
#                 layers.ReLU(),
#                 layers.Dropout(0.5), # dropout noise
#                 layers.Dense(num_classes, activation='linear'),
#                 layers.BatchNormalization(),
#             ]
#         )
    
#     # @tf.function
#     def call(self, x, noise=False, training=True):
#         h = x
#         for i in range(len(self.feature_nets)):
#             if noise:
#                 # Gaussian noise
#                 epsilon = tf.random.normal(shape=tf.shape(h), stddev=self.stddev)
#                 h += epsilon
#             h = self.feature_nets[i](h, training=training)
#         h = self.nets(h, training=training)
#         return tf.nn.softmax(h, axis=-1)
#%%