#%%
import argparse
import os

os.chdir(r'D:\EXoN_official') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/semi/proposal') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import yaml
import io
import matplotlib.pyplot as plt
from PIL import Image

from preprocess import fetch_dataset
from model import MixtureVAE
from criterion import ELBO_criterion
from mixup import augment, label_smoothing, non_smooth_mixup, weight_decay_decoupled
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset used for training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=100, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=100, 
                        help='number labeled examples (default: 100')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    parser.add_argument('--augment', default=False, type=bool,
                        help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    parser.add_argument('--bce', "--bce_reconstruction", default=True, type=bool,
                        help="Do BCE Reconstruction")
    parser.add_argument('--beta_trainable', default=True, type=bool,
                        help="trainable beta")
    # parser.add_argument('--depth', type=int, default=28, 
    #                     help='depth for WideResnet (default: 28)')
    # parser.add_argument('--width', type=int, default=2, 
    #                     help='widen factor for WideResnet (default: 2)')
    # parser.add_argument('--slope', type=float, default=0.1, 
    #                     help='slope parameter for LeakyReLU (default: 0.1)')
    # parser.add_argument('-dr', '--drop_rate', default=0, type=float, 
    #                     help='drop rate for the network')
    # parser.add_argument("-s", "--x_sigma", default=0.5, type=float,
    #                     help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=2, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma', default=4, type=float,  
                        help='variance of prior mixture component')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--kl_y_threshold', default=2.3, type=float,  
                        help='mutual information bound of discrete kl-divergence')
    parser.add_argument('--lambda1',default=1000, type=int, 
                        help='the weight of classification loss term')
    parser.add_argument('--lambda2',default=4, type=int, 
                        help='the weight of beta penalty term')
    parser.add_argument('--mixup_max_y', default=10, type=float, 
                        help='the max value for mixup(y) weight')
    parser.add_argument('--mixup_epoch_y',default=50, type=int, 
                        help='the max epoch to adjust mixup')
    # parser.add_argument('--recon_max', default=1, type=float, 
    #                     help='the max value for reconstruction error')
    # parser.add_argument('--recon_max_epoch',default=1, type=int, 
    #                     help='the max epoch to adjust reconstruction error')
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-ad', "--adjust_lr", default=[75, 90], type=arg_as_list,
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    # parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float)

    '''Optimizer Transport Estimation Parameters'''
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="the label smoothing epsilon for labeled data")

    '''Configuration'''
    parser.add_argument('--config_path', type=str, default=None, 
                        help='path to yaml config file, overwrites args')

    return parser
#%%
def load_config(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args['config_path'])    
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args
#%%
args = vars(get_args().parse_args(args=['--config_path', 'configs/mnist_100.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)

model_path = log_path + '/20220221-210550'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
model = MixtureVAE(args,
                num_classes,
                latent_dim=args['latent_dim'])
model.build(input_shape=(None, 28, 28, 1))
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
'''prior design'''
r = 2. * np.sqrt(args['sigma']) / np.sin(np.pi / 10.)
prior_means = np.array([[r*np.cos(np.pi/10), r*np.sin(np.pi/10)],
                        [r*np.cos(3*np.pi/10), r*np.sin(3*np.pi/10)],
                        [r*np.cos(5*np.pi/10), r*np.sin(5*np.pi/10)],
                        [r*np.cos(7*np.pi/10), r*np.sin(7*np.pi/10)],
                        [r*np.cos(9*np.pi/10), r*np.sin(9*np.pi/10)],
                        [r*np.cos(11*np.pi/10), r*np.sin(11*np.pi/10)],
                        [r*np.cos(13*np.pi/10), r*np.sin(13*np.pi/10)],
                        [r*np.cos(15*np.pi/10), r*np.sin(15*np.pi/10)],
                        [r*np.cos(17*np.pi/10), r*np.sin(17*np.pi/10)],
                        [r*np.cos(19*np.pi/10), r*np.sin(19*np.pi/10)]])
prior_means = tf.cast(prior_means[np.newaxis, :, :], tf.float32)
sigma = tf.cast(args['sigma'], tf.float32)
#%%
autotune = tf.data.AUTOTUNE
batch = lambda dataset: dataset.batch(batch_size=args['batch_size'], drop_remainder=False).prefetch(autotune)
iterator_test = iter(batch(test_dataset))
total_length = sum(1 for _ in test_dataset)
iteration = total_length // args['batch_size'] 
#%%
means = []
logvars = []
probs = []
labels = []
z_tildes = []
for i in tqdm.tqdm(range(iteration + 1)):
    image, label = next(iterator_test)
    mean, logvar, prob, y, z, z_tilde = model.encode(image, training=False)
    means.extend(mean)
    logvars.extend(logvar)
    probs.extend(prob)
    labels.extend(label)
    z_tildes.extend(z_tilde)
means = tf.stack(means, axis=0)
logvars = tf.stack(logvars, axis=0)
probs = tf.stack(probs, axis=0)
labels = tf.stack(labels, axis=0)
z_tildes = tf.stack(z_tildes, axis=0)
#%%
'''classification loss'''
classification_error = np.sum((tf.argmax(labels, axis=1) - tf.argmax(probs, axis=1) != 0).numpy()) / total_length
print('classification loss: ', classification_error)

'''KL divergence'''
kl1 = tf.reduce_mean(tf.reduce_sum(probs * (tf.math.log(tf.clip_by_value(probs, 1e-10, 1.)) + 
                                            tf.math.log(tf.cast(num_classes, tf.float32))), axis=1))
kl2 = tf.reduce_mean(tf.reduce_sum(tf.multiply(probs, 
                                                tf.reduce_sum(0.5 * (tf.math.pow(means - prior_means, 2) / sigma
                                                                    - 1
                                                                    + tf.math.log(sigma)
                                                                    + tf.math.exp(logvars) / sigma
                                                                    - logvars), axis=-1)), axis=-1))
print('KL divergence: ', (kl1 + kl2).numpy())
#%%
'''grid points'''
a = np.arange(-15, 15.1, 2.5)
b = np.arange(-15, 15.1, 2.5)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))
#%%
'''Figure 2 top panel'''
zmat = np.array(z_tildes)
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(zmat[:, 0], zmat[:, 1], c=tf.argmax(labels, axis=1).numpy(), s=10, cmap=plt.cm.Reds, alpha=1)
plt.savefig('./{}/latent.png'.format(model_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Figure 2 bottom panel'''
grid_output = model.decode(tf.cast(np.array(grid), tf.float32), training=False)
grid_output = grid_output.numpy()
plt.figure(figsize=(10, 10))
for i in range(len(grid)):
    plt.subplot(len(b), len(a), i+1)
    plt.imshow(grid_output[i].reshape(28, 28), cmap='gray_r')    
    plt.axis('off')
    plt.tight_layout() 
plt.savefig('./{}/reconstruction.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()

reconstruction = Image.open('./{}/reconstruction.png'.format(model_path))

plt.figure(figsize=(10, 10))
plt.xticks(np.arange(-15, 15.1, 5))    
plt.yticks(np.arange(-15, 15.1, 5))    
plt.tick_params(labelsize=30)    
plt.imshow(reconstruction, extent=[-16.3, 16.3, -16.3, 16.3])
plt.tight_layout() 
plt.savefig('./{}/reconstruction.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Appendix: Figure 1 middle panel'''
a = np.arange(-20, 20.1, 0.25)
b = np.arange(-20, 20.1, 0.25)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))
grid = tf.cast(np.array(grid), tf.float32)
grid_output = model.decode(grid, training=False)
grid_output = grid_output.numpy()
grid_prob = model.classify(grid_output, training=False)
grid_prob_argmax = np.argmax(grid_prob.numpy(), axis=1)
plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(grid[:, 0], grid[:, 1], c=grid_prob_argmax, s=10, cmap=plt.cm.Reds, alpha=1)
plt.savefig('./{}/conditional_prob.png'.format(model_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
# '''negative SSIM'''
# a = np.arange(-15, 15.1, 1.0)
# b = np.arange(-15, 15.1, 1.0)
# aa, bb = np.meshgrid(a, b, sparse=True)
# grid = []
# for b_ in reversed(bb[:, 0]):
#     for a_ in aa[0, :]:
#         grid.append(np.array([a_, b_]))
# grid_output = model.decoder(tf.cast(np.array(grid), tf.float32))
# ssim = 0
# for i in tqdm(range(len(grid_output))):
#     s = tf.image.ssim(tf.reshape(grid_output[i, :], (28, 28, 1)), tf.reshape(grid_output, (len(grid_output), 28, 28, 1)), 
#                     max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
#     ssim += np.sum(s.numpy())
# neg_ssim = (1 - ssim / (len(grid_output)*len(grid_output))) / 2
# print('negative SSIM: ', neg_ssim)
# result_negssim[str(PARAMS['lambda2'])] = result_negssim[str(PARAMS['lambda2'])] + [neg_ssim]
#%%    