#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
import numpy as np
#%%
class Encoder(K.models.Model):
    def __init__(self, latent_dim, num_classes, name="Encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.feature_num = 32
        self.nChannels = [self.feature_num * d for d in [1, 2, 4, 8]]
        
        self.net = K.Sequential(
            [
                layers.Conv2D(filters=self.nChannels[0], kernel_size=5, strides=2, padding='same'), # 16x16
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[1], kernel_size=4, strides=2, padding='same'), # 8x8
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[2], kernel_size=4, strides=2, padding='same'), # 4x4
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=self.nChannels[3], kernel_size=4, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Flatten(),
                layers.Dense(1024),
                layers.BatchNormalization(),
            ]
        )
        
        self.mean_layer = [layers.Dense(latent_dim, activation='linear') for _ in range(num_classes)]
        self.logvar_layer = [layers.Dense(latent_dim, activation='linear') for _ in range(num_classes)]
    
    @tf.function
    def call(self, x, training=True):
        h = self.net(x, training=training)
        mean = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.mean_layer])
        logvar = layers.Concatenate(axis=1)([d(h)[:, tf.newaxis, :] for d in self.logvar_layer])
        return mean, logvar
#%%
class Classifier(K.models.Model):
    def __init__(self, 
                 num_classes,
                 dropratio=0.1,
                 name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.units = K.Sequential(
            [
                layers.Conv2D(filters=128, kernel_size=4, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=128, kernel_size=4, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=dropratio),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                layers.SpatialDropout2D(rate=dropratio),
                
                layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.1),
                
                layers.GlobalAveragePooling2D(),
                
                layers.Dense(num_classes, activation='softmax')
            ]
        )
        self.bn = layers.BatchNormalization()
    
    @tf.function
    def call(self, x, training=True):
        h = self.units(x, training=training)
        return h
#%%
# classifier = Classifier(10, 0.1)
# classifier.build(input_shape=(None, 32, 32, 3))
# classifier.summary()
#%%
class Decoder(K.models.Model):
    def __init__(self, channel, activation, name="Decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.feature_num = 32
        self.nChannels = [self.feature_num * d for d in [8, 4, 2, 1]]
        
        self.net = K.Sequential(
            [
                layers.Dense(4*4*512),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Reshape((4, 4, 512)),
                
                layers.Conv2DTranspose(filters=self.nChannels[0], kernel_size=5, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=self.nChannels[1], kernel_size=5, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=self.nChannels[2], kernel_size=5, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2DTranspose(filters=self.nChannels[3], kernel_size=5, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                
                layers.Conv2D(filters=channel, kernel_size=4, strides=1, padding='same', activation=activation)
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
                 latent_dim=128, 
                 output_channel=3, 
                 dropratio=0.1,
                 activation='tanh',
                 input_dim=(None, 32, 32, 3), 
                 hard=True,
                 name='MixtureVAE', **kwargs):
        super(MixtureVAE, self).__init__(name=name, **kwargs)
        self.hard = hard
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.encoder = Encoder(latent_dim, num_classes)
        # self.FeatureExtractor = WideResNet(self.num_classes, args['depth'], args['width'], args['slope'], self.input_dim)
        # self.prob_layer = layers.Dense(num_classes, activation='softmax') 
        self.classifier = Classifier(num_classes, dropratio)
        self.decoder = Decoder(output_channel, activation)
    
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
        mean, logvar = self.encoder(x, training=training)
        epsilon = tf.random.normal((tf.shape(x)[0], self.num_classes, self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        # h = self.FeatureExtractor(x, training=training)
        # prob = self.prob_layer(h)
        prob = self.classifier(x, training=training)
        y = self.gumbel_max_sample(prob)
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        return mean, logvar, prob, y, z, z_tilde
    
    def classify(self, x, training=True):
        # h = self.FeatureExtractor(x, training=training)
        # prob = self.prob_layer(h)
        prob = self.classifier(x, training=training)
        return prob
    
    def decode(self, z, training=True):
        xhat = self.decoder(z, training=training) 
        return xhat
        
    @tf.function
    def call(self, x, training=True):
        mean, logvar = self.encoder(x, training=training)
        epsilon = tf.random.normal((tf.shape(x)[0], self.num_classes, self.latent_dim))
        z = mean + tf.math.exp(logvar / 2.) * epsilon 
        # assert z.shape == (tf.shape(x)[0], self.num_classes, self.latent_dim)
        
        # h = self.FeatureExtractor(x, training=training)
        # prob = self.prob_layer(h)
        prob = self.classifier(x, training=training)
        y = self.gumbel_max_sample(prob)
        # assert y.shape == (tf.shape(x)[0], self.num_classes)
        
        z_tilde = tf.squeeze(tf.matmul(y[:, tf.newaxis, :], z), axis=1)
        # assert z_tilde.shape == (tf.shape(x)[0], self.latent_dim)
        
        xhat = self.decoder(z_tilde, training=training) 
        # assert xhat.shape == (tf.shape(x)[0], self.input_dim[1], self.input_dim[2], self.input_dim[3])
        
        return mean, logvar, prob, y, z, z_tilde, xhat
#%%
# class ResidualUnit(K.layers.Layer):
#     def __init__(self, 
#                  filter_in, 
#                  filter_out,
#                  strides, 
#                  slope=0.1,
#                  **kwargs):
#         super(ResidualUnit, self).__init__(**kwargs)
        
#         self.norm1 = layers.BatchNormalization()
#         self.relu1 = layers.LeakyReLU(alpha=slope)
#         self.conv1 = layers.Conv2D(filters=filter_out, kernel_size=3, strides=strides, 
#                                     padding='same', use_bias=False)
#         self.norm2 = layers.BatchNormalization()
#         self.relu2 = layers.LeakyReLU(alpha=slope)
#         self.conv2 = layers.Conv2D(filters=filter_out, kernel_size=3, strides=1, 
#                                     padding='same', use_bias=False)
        
#         self.downsample = (filter_in != filter_out)
#         if self.downsample:
#             self.shortcut = layers.Conv2D(filters=filter_out, kernel_size=1, strides=strides, 
#                                         padding='same', use_bias=False)

#     @tf.function
#     def call(self, x, training=True):
#         if self.downsample:
#             x = self.relu1(self.norm1(x, training=training))
#             h = self.relu2(self.norm2(self.conv1(x), training=training))
#         else:
#             h = self.relu1(self.norm1(x, training=training))
#             h = self.relu2(self.norm2(self.conv1(h), training=training))
#         h = self.conv2(h)
#         if self.downsample:
#             h = h + self.shortcut(x)
#         else:
#             h = h + x
#         return h
# #%%
# class ResidualBlock(K.layers.Layer):
#     def __init__(self,
#                  n_units,
#                  filter_in,
#                  filter_out,
#                  unit,
#                  strides, 
#                  **kwargs):
#         super(ResidualBlock, self).__init__(**kwargs)
#         self.units = self._build_unit(n_units, unit, filter_in, filter_out, strides)
    
#     def _build_unit(self, n_units, unit, filter_in, filter_out, strides):
#         units = []
#         for i in range(n_units):
#             units.append(unit(filter_in if i == 0 else filter_out, filter_out, strides if i == 0 else 1))
#         return K.models.Sequential(units)
    
#     @tf.function
#     def call(self, x, training=True):
#         x = self.units(x, training=training)
#         return x
# #%%
# class WideResNet(K.models.Model):
#     def __init__(self, 
#                  num_classes,
#                  depth=28,
#                  width=2,
#                  slope=0.1,
#                  input_shape=(None, 32, 32, 3),
#                  name="WideResNet", 
#                  **kwargs):
#         super(WideResNet, self).__init__(input_shape, name=name, **kwargs)
        
#         assert (depth - 4) % 6 == 0
#         self.n_units = (depth - 4) // 6
#         self.nChannels = [16, 16*width, 32*width, 64*width]
#         self.slope = slope
        
#         self.conv = layers.Conv2D(filters=self.nChannels[0], kernel_size=3, strides=1, 
#                                     padding='same', use_bias=False)
        
#         self.block1 = ResidualBlock(self.n_units, self.nChannels[0], self.nChannels[1], ResidualUnit, 1)
#         self.block2 = ResidualBlock(self.n_units, self.nChannels[1], self.nChannels[2], ResidualUnit, 2)
#         self.block3 = ResidualBlock(self.n_units, self.nChannels[2], self.nChannels[3], ResidualUnit, 2)
        
#         self.norm = layers.BatchNormalization()
#         self.relu = layers.LeakyReLU(alpha=slope)
#         self.pooling = layers.GlobalAveragePooling2D()
#         self.dense = layers.Dense(num_classes)
    
#     @tf.function
#     def call(self, x, training=True):
#         h = self.conv(x)
#         h = self.block1(h, training=training)
#         h = self.block2(h, training=training)
#         h = self.block3(h, training=training)
#         h = self.relu(self.norm(h, training=training))
#         h = self.pooling(h)
#         h = self.dense(h)
#         return h
#%%