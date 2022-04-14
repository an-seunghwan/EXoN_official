#%%
import argparse
import os

os.chdir(r'D:\EXoN_official') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/semi/proposal') # main directory (repository)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import yaml
import io
import matplotlib.pyplot as plt
from PIL import Image

from preprocess import fetch_dataset
from model2 import MixtureVAE
from criterion1 import ELBO_criterion
# from mixup import augment, label_smoothing, non_smooth_mixup, weight_decay_decoupled
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
                        help='number labeled examples (default: 100), all labels are balanced')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    parser.add_argument('--augment', default=True, type=bool,
                        help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    parser.add_argument('--bce', "--bce_reconstruction", default=False, type=bool,
                        help="Do BCE Reconstruction")
    parser.add_argument('--beta_trainable', default=False, type=bool,
                        help="trainable beta")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=2, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma', default=4, type=float,  
                        help='variance of prior mixture component')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--kl_y_threshold', default=0, type=float,  
                        help='mutual information bound of discrete kl-divergence')
    parser.add_argument('--lambda1', default=6000, type=int, # labeled dataset ratio?
                        help='the weight of classification loss term')
    '''lambda2 -> beta'''
    parser.add_argument('--lambda2', default=10, type=int, 
                        help='the weight of beta penalty term, initial value of beta')
    parser.add_argument('--rampup_epoch',default=10, type=int, 
                        help='the max epoch to adjust learning rate and unsupervised weight')
    parser.add_argument('--rampdown_epoch',default=10, type=int, 
                        help='the last epoch to adjust learning rate')
    parser.add_argument('--entropy_loss', default=False, type=bool,
                        help="add entropy minimization regularization to loss")
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=3e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float)
    # parser.add_argument('--clipnorm', default=1, type=float)

    '''Interpolation Parameters'''
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

# model_path = log_path + '/20220310-125521'
model_path = log_path + '/beta_{}'.format(0.25)
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
print('test dataset classification error: ', classification_error)
#%%
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
'''interpolation on latent space'''
z_inter = (prior_means.numpy()[0][0], prior_means.numpy()[0][1])    

np.random.seed(1)
samples = []
color = []
for i in range(num_classes):
    samples.extend(np.random.multivariate_normal(mean=prior_means.numpy()[0][i, :2], cov=np.array([[sigma.numpy(), 0], 
                                                                                        [0, sigma.numpy()]]), size=1000))
    color.extend([i] * 1000)
samples = np.array(samples)

plt.figure(figsize=(10, 10))
plt.tick_params(labelsize=30)    
plt.locator_params(axis='y', nbins=8)
plt.scatter(zmat[:, 0], zmat[:, 1], c=tf.argmax(labels, axis=1).numpy(), s=10, cmap=plt.cm.Reds, alpha=1)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.scatter(z_inter[0][0], z_inter[0][1], color='blue', s=100)
plt.annotate('A', (z_inter[0][0], z_inter[0][1]), fontsize=30)
plt.scatter(z_inter[1][0], z_inter[1][1], color='blue', s=100)
plt.annotate('B', (z_inter[1][0], z_inter[1][1]), fontsize=30)
plt.plot((z_inter[0][0], z_inter[1][0]), (z_inter[0][1], z_inter[1][1]), color='black', linewidth=2, linestyle='--')
plt.xlabel("$z_0$", fontsize=30)
plt.ylabel("$z_1$", fontsize=30)
plt.savefig('./{}/interpolation_path.png'.format(model_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
#%%
'''
negative SSIM
'''
a = np.arange(-15, 15.1, 1.0)
b = np.arange(-15, 15.1, 1.0)
aa, bb = np.meshgrid(a, b, sparse=True)
grid = []
for b_ in reversed(bb[:, 0]):
    for a_ in aa[0, :]:
        grid.append(np.array([a_, b_]))
grid_output = model.decoder(tf.cast(np.array(grid), tf.float32))
ssim = 0
for i in tqdm.tqdm(range(len(grid_output))):
    s = tf.image.ssim(tf.reshape(grid_output[i, :], (28, 28, 1)), tf.reshape(grid_output, (len(grid_output), 28, 28, 1)), 
                    max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim += np.sum(s.numpy())
neg_ssim = (1 - ssim / (len(grid_output)*len(grid_output))) / 2
print('negative SSIM: ', neg_ssim)
#%%
with open('{}/result.txt'.format(model_path), "w") as file:
    file.write('TEST classification error: {:.3f}%\n\n'.format(classification_error * 100))
    file.write('KL-divergence: {:.3f}\n\n'.format((kl1 + kl2).numpy()))
    file.write('negative SSIM: {:.3f}\n\n'.format(neg_ssim))
#%%
# l = 10
# model_path = log_path + '/path (consistency_interpolation)/lambda2_{}'.format(l)
# model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
# model = MixtureVAE(args,
#                 num_classes,
#                 latent_dim=args['latent_dim'])
# model.build(input_shape=(None, 28, 28, 1))
# model.load_weights(model_path + '/' + model_name)
#%%
'''interpolation'''
inter = np.linspace(z_inter[0], z_inter[1], 10)
inter_recon = model.decoder(inter)

figure = plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10+1, i+1)
    plt.imshow(inter_recon[i].numpy().reshape(28, 28), cmap='gray_r')
    plt.axis('off')
plt.savefig('./{}/interpolation_path_recon.png'.format(model_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
#%%
'''interpolation path and reconstruction'''
betas = [0.1, 0.2, 0.25, 0.5, 0.75, 1, 5, 10, 50]
for l in betas:
    img = [Image.open('./logs/mnist_100/beta_{}/interpolation_path.png'.format(l)),
            Image.open('./logs/mnist_100/beta_{}/interpolation_path_recon.png'.format(l))]

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 0.25]})
    a0.imshow(img[0])    
    a0.axis('off')
    a1.imshow(img[1])    
    a1.axis('off')
    plt.tight_layout() 
    plt.savefig('./logs/mnist_100/beta_{}/interpolation_path_and_recon.png'.format(l),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()
#%%
'''path: latent space and reconstruction'''
betas = [0.1, 0.5, 5, 50]
img = []
for l in betas:
    img.append([Image.open('./logs/mnist_100/beta_{}/latent.png'.format(l)),
                Image.open('./logs/mnist_100/beta_{}/reconstruction.png'.format(l))])

plt.figure(figsize=(10, 5))
for i in range(len(img)):
    plt.subplot(2, len(betas), i+1)
    plt.imshow(img[i][0])    
    plt.axis('off')
    plt.tight_layout() 
    # plt.title('$\lambda_1=6000$, $\\beta={}$'.format(betas[i]))
    plt.title('$\\beta={}$'.format(betas[i]))
    
    plt.subplot(2, len(betas), i+len(img)+1)
    plt.imshow(img[i][1])    
    plt.axis('off')
    plt.tight_layout() 
    
plt.savefig('./logs/mnist_100/path_latent_recon.png',
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''path: test classifiation error'''
betas = [0.1, 0.25, 0.5, 0.75, 1, 5, 10, 50]
errors = {}
kls = {}
ssims = {}
for b in betas:
    model_path = log_path + '/beta_{}'.format(b)
    with open('{}/result.txt'.format(model_path), 'r') as f:
        line = f.readlines()
    errors[b] = line[0].split(' ')[-1][:-2]
    kls[b] = line[2].split(' ')[-1][:-1]
    ssims[b] = line[4].split(' ')[-1][:-1]
pd.DataFrame.from_dict(errors, orient='index').rename(columns={0: 'test error'}).to_csv(log_path + '/test_error_path.csv')
pd.DataFrame.from_dict(kls, orient='index').rename(columns={0: 'KL'}).to_csv(log_path + '/kl_path.csv')
pd.DataFrame.from_dict(ssims, orient='index').rename(columns={0: 'negative SSIM'}).to_csv(log_path + '/negative_ssim_path.csv')
#%%