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
import seaborn as sns

from preprocess import fetch_dataset
from model import MixtureVAE
from criterion import ELBO_criterion
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

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cmnist, cifar10, svhn, svhn+extra)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=400, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    parser.add_argument('--augment', default=True, type=bool,
                        help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    parser.add_argument('-dr', '--drop_rate', default=0, type=float, 
                        help='drop rate for the network')
    parser.add_argument('--bce', "--bce_reconstruction", default=False, type=bool,
                        help="Do BCE Reconstruction")
    parser.add_argument('--beta_trainable', default=False, type=bool,
                        help="trainable beta")
    # parser.add_argument('--depth', type=int, default=28, 
    #                     help='depth for WideResnet (default: 28)')
    # parser.add_argument('--width', type=int, default=2, 
    #                     help='widen factor for WideResnet (default: 2)')
    # parser.add_argument('--slope', type=float, default=0.1, 
    #                     help='slope parameter for LeakyReLU (default: 0.1)')

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=256, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma1', default=0.1, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--sigma2', default=1, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--dist', default=1, type=float,  
                        help='first 10-dimension latent mean vector value')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--kl_y_threshold', default=0, type=float,  
                        help='mutual information bound of discrete kl-divergence')
    parser.add_argument('--lambda1', default=5000, type=int, # labeled dataset ratio?
                        help='the weight of classification loss term')
    '''lambda2 -> beta'''
    parser.add_argument('--lambda2', default=0.01, type=int, 
                        help='the weight of beta penalty term, initial value of beta')
    parser.add_argument('--rampup_epoch', default=50, type=int, 
                        help='the max epoch to adjust unsupervised weight')
    # parser.add_argument('--rampdown_epoch', default=50, type=int, 
    #                     help='the last epoch to adjust learning rate')
    parser.add_argument('--entropy_loss', default=True, type=bool,
                        help="add entropy minimization regularization to loss")
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-ad', "--adjust_lr", default=[250, 350], type=arg_as_list, # classifier optimizer scheduling
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.5, type=float)
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float)

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
args = vars(get_args().parse_args(args=['--config_path', 'configs/cifar10_4000.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)

model_path = log_path + '/20220305-014914'
model_name = [x for x in os.listdir(model_path) if x.endswith('.h5')][0]
model = MixtureVAE(args,
            num_classes,
            latent_dim=args['latent_dim'],
            dropratio=args['drop_rate'])
model.build(input_shape=(None, 32, 32, 3))
model.load_weights(model_path + '/' + model_name)
model.summary()
#%%
classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classdict = {i:x for i,x in enumerate(classnames)}
#%%
'''prior design'''
prior_means = np.zeros((num_classes, args['latent_dim']))
prior_means[:, :num_classes] = np.eye(num_classes) * args['dist']
prior_means = tf.cast(prior_means[np.newaxis, :, :], tf.float32)

sigma_vector = np.ones((1, args['latent_dim'])) 
sigma_vector[0, :num_classes] = args['sigma1']
sigma_vector[0, num_classes:] = args['sigma2']
sigma_vector = tf.cast(sigma_vector, tf.float32)
#%%
'''test dataset classification error'''
autotune = tf.data.AUTOTUNE
batch = lambda dataset: dataset.batch(batch_size=args['batch_size'], drop_remainder=False).prefetch(autotune)
iterator_test = iter(batch(test_dataset))
total_length = sum(1 for _ in test_dataset)
iteration = total_length // args['batch_size'] 

error_count = 0
for x_test_batch, y_test_batch in batch(test_dataset):
    _, _, prob, _, _, _, _ = model(x_test_batch, training=False)
    error_count += np.sum(tf.argmax(prob, axis=-1).numpy() - tf.argmax(y_test_batch, axis=-1).numpy() != 0)
print('TEST classification error: {:.2f}%'.format(error_count / total_length * 100))
#%%
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5
from tensorflow.keras.utils import to_categorical
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args['batch_size'])
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args['batch_size'])
#%%
'''V-nat'''
var_list = []
for k in range(num_classes):
    x = x_test[np.where(y_test == k)[0]]
    mean, logvar, _, _, _, _, _ = model(x, training=False)  
    var = np.exp(logvar.numpy())
    var_list.append(var[:, k, :])
var_list = np.array(var_list)

V_nat = np.log(np.mean(sigma_vector / var_list, axis=1))

k = 1
delta = 1. 

print('cardinality of activated latent subspace:', sum(V_nat[k] > delta))

plt.figure(figsize=(7, 3))
plt.bar(np.arange(args['latent_dim']), V_nat[k], width=2) 
plt.xlabel("latent dimensions", size=16)
plt.ylabel("V-nat", size=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.savefig('{}/vnat.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
plt.figure(figsize=(7, 3))
plt.bar(np.arange(args['latent_dim']), np.sort(V_nat[k]), width=2) 
plt.xlabel("latent dimensions", size=16)
plt.ylabel("V-nat", size=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.savefig('{}/vnat_sorted.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Appendix: Figure 4'''
colors = plt.rcParams["axes.prop_cycle"]()
fig, axes = plt.subplots(10, 1, sharex=True, sharey=True, figsize=(10, 15))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
for k in range(num_classes):
    c = next(colors)["color"]
    axes.flatten()[k].bar(np.arange(args['latent_dim']), V_nat[k], 
                          color=c, width=2, label='{}'.format(classdict.get(k))) 
    axes.flatten()[k].legend(loc='upper left', fontsize=15)
    axes.flatten()[k].tick_params(labelsize=15)
plt.xlabel("latent dimensions", size=17)
plt.ylabel("V-nat", size=17)
plt.savefig('{}/all_vnat.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Appendix: Figure 5'''
df_vnat = pd.DataFrame(V_nat.T, columns=list(classdict.values()))
corr = df_vnat.corr()

plt.subplots(figsize=(7, 7))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, 
            cmap='RdYlBu_r', 
            annot=True,   
            fmt='.3f',
            annot_kws={"size": 9}, 
            mask=mask,      
            linewidths=.5,  
            cbar_kws={"shrink": .5},
            vmin=-1, vmax=1)  
plt.savefig('{}/vnat_corr.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
data_dir = r'D:\cifar10_{}'.format(5000)
idx = np.arange(100)
x = np.array([np.load(data_dir + '/x_{}.npy'.format(i)) for i in idx])
y = np.array([np.load(data_dir + '/y_{}.npy'.format(i)) for i in idx])
x = (tf.cast(x, tf.float32) - 127.5) / 127.5

_, _, _, _, _, z, images = model(x, training=False)  
#%%
'''Figure 2'''
plt.figure(figsize=(15, 15))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow((images[i] + 1) / 2)
    plt.axis('off')
plt.tight_layout()
plt.savefig('{}/train_recon.png'.format(model_path))
plt.show()
plt.close()
#%%
'''test reconstruction'''
x = x_test[:49]

_, _, _, _, _, _, images = model(x, training=False)  
plt.figure(figsize=(15, 15))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow((images[i] + 1) / 2)
    plt.axis('off')
plt.tight_layout()
plt.savefig('{}/test_recon.png'.format(model_path))
plt.show()
plt.close()
#%%
'''Figure 6'''
x = np.load(data_dir + '/x_{}.npy'.format(7))
x = (tf.cast(x, tf.float32) - 127.5) / 127.5
mean, logvar, prob, _, _, latent, images = model(x[tf.newaxis, ...], training=False)
# mean = mean[0, np.argmax(prob), :]
latent = np.squeeze(latent)

from copy import deepcopy
noise_z = deepcopy(latent)
keep_z = deepcopy(latent)

tf.random.set_seed(520)
zeros = np.zeros((args['latent_dim'], ))

signal_noise = np.zeros((args['latent_dim'], ))
signal_noise[V_nat[k] > delta] = tf.random.uniform((sum(V_nat[k] > delta), ), -2, 2).numpy()

signal_keep = np.zeros((args['latent_dim'], ))
signal_keep[V_nat[k] <= delta] = tf.random.uniform((sum(V_nat[k] <= delta), ), -2, 2).numpy()

noise_z += signal_noise
keep_z += signal_keep

signals = [zeros, signal_noise, signal_keep]
perturbed_z = [latent, noise_z, keep_z]

clear_img = model.decode(perturbed_z[0][None, ...], training=False)[-1].numpy()
noise_img = model.decode(perturbed_z[1][None, ...], training=False)[-1].numpy()
keep_img = model.decode(perturbed_z[2][None, ...], training=False)[-1].numpy()

fig, axes = plt.subplots(2, 3, figsize=(15, 11))
axes.flatten()[0].plot(signals[0])
axes.flatten()[0].tick_params(labelsize=25)
axes.flatten()[0].set_xlabel('latent dimension', fontsize=24)
axes.flatten()[0].set_ylabel('noise', fontsize=25)
axes.flatten()[3].imshow((clear_img + 1.) / 2.)
axes.flatten()[3].axis('off')

axes.flatten()[1].bar(np.arange(args['latent_dim']), signals[1], width=2)
axes.flatten()[1].tick_params(labelsize=25)
axes.flatten()[1].set_xlabel('latent dimension', fontsize=24)
axes.flatten()[4].imshow((noise_img + 1.) / 2.)
axes.flatten()[4].axis('off')

axes.flatten()[2].bar(np.arange(args['latent_dim']), signals[2], width=2)
axes.flatten()[2].tick_params(labelsize=25)
axes.flatten()[2].set_xlabel('latent dimension', fontsize=24)
axes.flatten()[5].imshow((keep_img + 1.) / 2.)
axes.flatten()[5].axis('off')

plt.savefig('{}/blur.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Figure 7'''
fig, axes = plt.subplots(2, 10, figsize=(25, 5))
inter = np.linspace(z[7], z[43], 10)
inter_recon = model.decode(inter, training=False)
for i in range(10):
    axes.flatten()[i].imshow((inter_recon[i].numpy() + 1.) / 2.)
    axes.flatten()[i].axis('off')
    
inter = np.linspace(z[7], z[8], 10)
inter_recon = model.decode(inter, training=False)
for i in range(10):
    axes.flatten()[10 + i].imshow((inter_recon[i].numpy() + 1.) / 2.)
    axes.flatten()[10 + i].axis('off')
    
plt.savefig('{}/interpolation.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Figure 8'''
x = np.load(data_dir + '/x_{}.npy'.format(119))
x = (tf.cast(x, tf.float32) - 127.5) / 127.5
mean, logvar, prob, _, _, z, images = model(x[tf.newaxis, ...], training=False)
# mean = mean[0, np.argmax(prob), :]

clear_rand = deepcopy(z)
clear_rand = np.tile(clear_rand, (21, 1))

tf.random.set_seed(520)
clear_rand[:, V_nat[k] > delta] += tf.random.uniform((21, sum(V_nat[k] > delta)), -1.5, 1.5).numpy()
clear_rand[0, :] = z
clear_rand_recon = model.decode(clear_rand, training=False)
fig, axes = plt.subplots(3, 7, figsize=(10, 4))
for i in range(21):
    axes.flatten()[i].imshow((clear_rand_recon[i].numpy() + 1.) / 2.)
    axes.flatten()[i].axis('off')
plt.savefig('{}/blur_many.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Appendix: Figure 6'''
x = np.load(data_dir + '/x_{}.npy'.format(8))
x = (tf.cast(x, tf.float32) - 127.5) / 127.5
mean, logvar, prob, _, _, z, images = model(x[tf.newaxis, ...], training=False)
# mean = mean[0, np.argmax(prob), :]
k = np.argmax(prob)

sorted_idx = np.argsort(V_nat[k])[::-1][:sum(V_nat[k] > delta)]

fig, axes = plt.subplots(5, 11, figsize=(10, 4))
for j in range(5):
    one_rands = np.tile(z, (11, 1))
    one_rands[:, sorted_idx[j]] = 0
    one_rands[:, sorted_idx[j]] += np.round(np.linspace(-3, 3, 11), 2)
    one_rand_recon = model.decode(one_rands, training=False)
    for i in range(11):
        axes[j][i].imshow((one_rand_recon[i].numpy() + 1.) / 2.)
        axes[j][i].axis('off')
plt.savefig('{}/oneaxis.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''Figure 4 (path experiments)'''
# path = pd.read_csv('./assets/{}/path/path_{}.csv'.format('cifar10', '10000.0'))
# fig = plt.figure(figsize=(8, 4))
# ax = fig.add_subplot(111)
# plot1 = ax.plot(path['lambda2'], path['cardinalitypath'], label = '$|\mathcal{A}_k(1)|$')
# ax2 = ax.twinx()
# plot2 = ax2.plot(path['lambda2'], path['negssimpath'], linestyle='dashed', label = 'negative SSIM')
# plots = plot1 + plot2
# labs = [p.get_label() for p in plots]
# ax.legend(plots, labs, loc='upper right', fontsize=18)
# # ax.set_title('automobile', fontsize=16)
# ax.set_xlabel(r'$\beta$', fontsize=21)
# ax.set_ylabel('cardinality', fontsize=18)
# ax2.set_ylabel('negative SSIM', fontsize=18)
# ax.tick_params(labelsize=16)
# ax2.tick_params(labelsize=16)
# ax.locator_params(axis='y', nbins=6)
# ax2.locator_params(axis='y', nbins=5)
# plt.savefig('./assets/{}/path/lambda2_path.png'.format('cifar10'),
#             dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
# plt.close()
#%%
'''rebuttal'''
data_dir = r'D:\cifar10_{}'.format(5000)
idx = np.arange(100)
x = np.array([np.load(data_dir + '/x_{}.npy'.format(i)) for i in idx])
y = np.array([np.load(data_dir + '/y_{}.npy'.format(i)) for i in idx])
x = (tf.cast(x, tf.float32) - 127.5) / 127.5

_, _, _, _, _, z, images = model(x, training=False)  
#%%
'''interpolation'''
# class_idx = 1
# i = 0
# j = 5
# class_idx = 7
# i = 0
# j = 2
fig, axes = plt.subplots(2, 10, figsize=(25, 5))
for idx, (class_idx, i, j) in enumerate([[1, 0, 5], [7, 0, 2]]):
    interpolation_idx = np.where(np.argmax(y, axis=-1) == class_idx)[0]

    inter = np.linspace(z[interpolation_idx[i]], z[interpolation_idx[j]], 8)
    inter_recon = model.decode(inter, training=False)

    axes.flatten()[idx*10 + 0].imshow((x[interpolation_idx[i]] + 1.) / 2.)
    axes.flatten()[idx*10 + 0].axis('off')
    for i in range(8):
        axes.flatten()[idx*10 + i+1].imshow((inter_recon[i].numpy() + 1.) / 2.)
        axes.flatten()[idx*10 + i+1].axis('off')
    axes.flatten()[idx*10 + 9].imshow((x[interpolation_idx[j]] + 1.) / 2.)
    axes.flatten()[idx*10 + 9].axis('off')
plt.savefig('{}/interpolation.png'.format(model_path),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
'''negative SSIM'''
# from tensorflow.keras.datasets.cifar10 import load_data
# (x_train, y_train), (x_test, y_test) = load_data()
# #%%
# x = np.array([x_train[np.where(y_train == i)[0][:100]] for i in range(10)])
# x = np.reshape(x, (-1, 32, 32, 3))
# x = (tf.cast(x, tf.float32) - 127.5) / 127.5
# y = np.array([tf.one_hot([i] * 100, depth=10).numpy() for i in range(10)])
# y = np.reshape(y, (-1, 10))
# #%%
# mean, logvar, prob, _, _, z, images = model(x, training=False)
# #%%
# x = images
# ssim = 0
# for i in tqdm.tqdm(range(len(x))):
#     s = tf.image.ssim(tf.reshape(x[i, :], (32, 32, 3)), 
#                       tf.reshape(x, (len(x), 32, 32, 3)), 
#                     max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
#     ssim += np.sum(s.numpy())
# neg_ssim = (1 - ssim / (len(x)*len(x))) / 2
# print('negative SSIM: ', neg_ssim)
#%%