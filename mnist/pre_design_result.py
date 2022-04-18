#%%
import argparse
import os

os.chdir(r'D:\EXoN_official') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/EXoN_official') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import to_categorical
import tqdm
import yaml
import io
import matplotlib.pyplot as plt
from PIL import Image

# import datetime
# current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from model import MixtureVAE
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
                        help='dataset used for training (only mnist)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--labeled-batch-size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=20, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', default=5, type=int,
                        metavar='N', help='reconstruct frequency (default: 5)')
    parser.add_argument('--labeled_examples', type=int, default=20, 
                        help='number labeled examples (default: 20), all labels are balanced')
    parser.add_argument('--augment', default=True, type=bool,
                        help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    parser.add_argument('--bce_reconstruction', default=False, type=bool,
                        help="Do BCE Reconstruction")
    # parser.add_argument('--beta_trainable', default=False, type=bool,
    #                     help="trainable beta")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=2, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma', default=4, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--radius', default=4, type=float,  
                        help='center coordinate of pre-designed components: (-r, 0), (r, 0), control diversity')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--lambda1', default=6000, type=int, # labeled dataset ratio?
                        help='the weight of classification loss term')
    parser.add_argument('--beta', default=1, type=int, 
                        help='value of beta (observation noise)')
    parser.add_argument('--rampup_epoch',default=10, type=int, 
                        help='the max epoch to adjust learning rate and unsupervised weight')
    parser.add_argument('--rampdown_epoch',default=10, type=int, 
                        help='the last epoch to adjust learning rate')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=3e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    '''Interpolation Parameters'''
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="beta distribution parameter")

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
args = vars(get_args().parse_args(args=['--config_path', 'configs/mnist_pre_design.yaml']))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_pre_design'

'''dataset'''
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train[..., tf.newaxis]
x_test = (x_test.astype("float32") - 127.5) / 127.5
x_test = x_test[..., tf.newaxis]

# dataset only from label 0 and 1
label = np.array([0, 1])
num_classes = len(label)
train_idx = np.isin(y_train, label)
test_idx = np.isin(y_test, label)

x_train = x_train[train_idx]
y_train = y_train[train_idx]
x_test = x_test[test_idx]
y_test = y_test[test_idx]

np.random.seed(1)
# ensure that all classes are balanced
lidx = np.concatenate(
    [
        np.random.choice(
            np.where(y_train == i)[0],
            int(args["labeled_examples"] / num_classes),
            replace=False,
        )
        for i in [0, 1]
    ]
)
x_train_L = x_train[lidx]
y_train_L = to_categorical(y_train[lidx], num_classes=num_classes)

datasetL = tf.data.Dataset.from_tensor_slices((x_train_L, y_train_L))
datasetU = tf.data.Dataset.from_tensor_slices((x_train))
total_length = sum(1 for _ in datasetU)

save_path = f'{log_path}/radius_{args["radius"]}'
model_name = [x for x in os.listdir(save_path) if x.endswith('.h5')][0]
model = MixtureVAE(args,
                num_classes,
                latent_dim=args['latent_dim'])
model.build(input_shape=(None, 28, 28, 1))
model.load_weights(save_path + '/' + model_name)
model.summary()

'''prior design'''
prior_means = np.array([[-args['radius'], 0], [args['radius'], 0]])
prior_means = prior_means[np.newaxis, :, :]
prior_means = tf.cast(prior_means, tf.float32)
sigma = tf.cast(args['sigma'], tf.float32)

'''initialize beta'''
beta = tf.cast(args['beta'], tf.float32) 
#%%
"""reconstruction"""
grid = np.array([[n, 0] for n in np.linspace(-10, 10, 11)])
grid_output = model.decode(tf.cast(np.array(grid), tf.float32), training=False)
grid_output = (grid_output.numpy() + 1) / 2
plt.figure(figsize=(8, 4))
for i in range(len(grid)):
    plt.subplot(1, len(grid), i + 1)
    plt.imshow(grid_output[i].reshape(28, 28), cmap="gray_r")
    plt.axis("off")
    plt.tight_layout()
plt.savefig('./{}/reconstruction.png'.format(save_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
"""posterior"""
zmat = []
mean, logvar, prob, y, z, z_tilde, xhat = model(x_test, training=False)
zmat.extend(z_tilde.numpy().reshape(-1, args["latent_dim"]))
zmat = np.array(zmat)
plt.figure(figsize=(7, 4))
plt.tick_params(labelsize=10)
plt.scatter(zmat[:, 0], zmat[:, 1], c=y_test, s=10, alpha=1)
plt.savefig('./{}/latent_space.png'.format(save_path), 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%
"""Appendix: Figure 2"""
radius = [4, 8, 12, 16]
reconstruction = []
for r in radius:
    save_path = f'{log_path}/radius_{r}'
    reconstruction.append(
        Image.open(
            './{}/reconstruction.png'.format(save_path)
            )
        )
plt.figure(figsize=(10, 5))
for i in range(len(reconstruction)):
    plt.subplot(4, 1, i + 1)
    plt.imshow(reconstruction[i], cmap="gray_r")
    plt.axis("off")
    plt.tight_layout()
plt.savefig(f'./{log_path}/predesign_reconstruction.png', 
            dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()

reconstruction = Image.open(f'./{log_path}/predesign_reconstruction.png')

plt.figure(figsize=(10, 10))
plt.xticks(np.linspace(-10, 10, 11))
plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=0)
plt.imshow(reconstruction, extent=[-11.35, 11.2, -5, 5])
plt.tight_layout()
plt.savefig(f'./{log_path}/predesign_reconstruction.png',
        dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close()
#%%