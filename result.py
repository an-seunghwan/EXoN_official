#%%
import argparse
import os

os.chdir(r"D:\EXoN_official")  # main directory (repository)
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
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v
#%%
def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (only cifar10)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--batch_size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    # parser.add_argument('--labeled_batch_size', default=32, type=int,
    #                     metavar='N', help='mini-batch size (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    # parser.add_argument('--epochs', default=600, type=int, 
    #                     metavar='N', help='number of total epochs to run')
    # parser.add_argument('--start_epoch', default=0, type=int, 
    #                     metavar='N', help='manual epoch number (useful on restarts)')
    # parser.add_argument('--reconstruct_freq', default=10, type=int,
    #                     metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000), all labels are balanced')
    # parser.add_argument('--validation_examples', type=int, default=5000, 
    #                     help='number validation examples (default: 5000')
    # parser.add_argument('--augment', default=True, type=bool,
    #                     help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    parser.add_argument('--drop_rate', default=0, type=float, 
                        help='drop rate for the network')
    # parser.add_argument("--bce_reconstruction", default=False, type=bool,
    #                     help="Do BCE Reconstruction")
    # parser.add_argument('--beta_trainable', default=False, type=bool,
    #                     help="trainable beta")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', default=256, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma1', default=0.1, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--sigma2', default=1, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--dist', default=1, type=float,  
                        help='first 10-dimension latent mean vector value')

    # '''VAE Loss Function Parameters'''
    # parser.add_argument('--lambda1', default=5000, type=int, 
    #                     help='the weight of classification loss term')
    # parser.add_argument('--beta', default=0.01, type=int, 
    #                     help='value of observation noise')
    # parser.add_argument('--rampup_epoch', default=50, type=int, 
    #                     help='the max epoch to adjust unsupervised weight')
    
    # '''Optimizer Parameters'''
    # parser.add_argument('--learning_rate', default=0.001, type=float,
    #                     metavar='LR', help='initial learning rate')
    # parser.add_argument("--adjust_lr", default=[250, 350, 450], type=arg_as_list, # classifier optimizer scheduling
    #                     help="The milestone list for adjust learning rate")
    # parser.add_argument('--lr_gamma', default=0.5, type=float)
    # parser.add_argument('--weight_decay', default=5e-4, type=float)
    # parser.add_argument('--epsilon', default=0.1, type=float,
    #                     help="beta distribution parameter")

    '''Configuration'''
    parser.add_argument('--config_path', type=str, default=None, 
                        help='path to yaml config file, overwrites args')

    return parser
#%%
def load_config(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args["config_path"])
    with open(config_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args
#%%
args = vars(get_args().parse_args(args=["--config_path", "configs/cifar10_4000.yaml"]))

dir_path = os.path.dirname(os.path.realpath(__file__))
if args["config_path"] is not None and os.path.exists(os.path.join(dir_path, args["config_path"])):
    args = load_config(args)

log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(
    args, log_path
)

# model_path = log_path + '/20220419-194055'
model_path = log_path + "/beta_{}".format(0.05)
model_name = [x for x in os.listdir(model_path) if x.endswith(".h5")][0]
model = MixtureVAE(
    args, num_classes, latent_dim=args["latent_dim"], dropratio=args["drop_rate"]
)
model.build(input_shape=(None, 32, 32, 3))
model.load_weights(model_path + "/" + model_name)
model.summary()
#%%
classnames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
classdict = {i: x for i, x in enumerate(classnames)}
#%%
"""prior design"""
prior_means = np.zeros((num_classes, args["latent_dim"]))
prior_means[:, :num_classes] = np.eye(num_classes) * args["dist"]
prior_means = tf.cast(prior_means[np.newaxis, :, :], tf.float32)

sigma_vector = np.ones((1, args["latent_dim"]))
sigma_vector[0, :num_classes] = args["sigma1"]
sigma_vector[0, num_classes:] = args["sigma2"]
sigma_vector = tf.cast(sigma_vector, tf.float32)
#%%
"""Test dataset classification error"""
autotune = tf.data.AUTOTUNE
batch = lambda dataset: dataset.batch(
    batch_size=args["batch_size"], drop_remainder=False
).prefetch(autotune)
# iterator_test = iter(batch(test_dataset))
total_length = sum(1 for _ in test_dataset)
iteration = total_length // args["batch_size"]

error_count = 0
for x_test_batch, y_test_batch in batch(test_dataset):
    _, _, prob, _, _, _, _ = model(x_test_batch, training=False)
    error_count += np.sum(
        tf.argmax(prob, axis=-1).numpy() - tf.argmax(y_test_batch, axis=-1).numpy() != 0
    )
print("TEST classification error: {:.2f}%".format(error_count / total_length * 100))
#%%
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_test = (x_test.astype("float32") - 127.5) / 127.5
from tensorflow.keras.utils import to_categorical

y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
    args["batch_size"]
)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
    args["batch_size"]
)
#%%
"""Figure 5(a): V-nat"""
var_list = []
for k in range(num_classes):
    x = x_test[np.where(y_test == k)[0]]
    mean, logvar, _, _, _, _, _ = model(x, training=False)
    var = np.exp(logvar.numpy())
    var_list.append(var[:, k, :])
var_list = np.array(var_list)

V_nat = np.log(np.mean(sigma_vector / var_list, axis=1))

k = 1
delta = 1

print("cardinality of activated latent subspace:", sum(V_nat[k] > delta))

plt.figure(figsize=(10, 1.5))
plt.bar(np.arange(args["latent_dim"]), V_nat[k], width=2)
plt.xlabel("latent dimensions", size=16)
plt.ylabel("V-nat", size=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis="y", nbins=3)
plt.savefig(
    "{}/vnat.png".format(model_path), dpi=100, bbox_inches="tight", pad_inches=0.1
)
plt.show()
plt.close()
#%%
plt.figure(figsize=(7, 3))
plt.bar(np.arange(args["latent_dim"]), np.sort(V_nat[k]), width=2)
plt.xlabel("latent dimensions", size=16)
plt.ylabel("V-nat", size=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis="y", nbins=6)
plt.savefig(
    "{}/vnat_sorted.png".format(model_path),
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()
plt.close()
#%%
"""Appendix Figure 2"""
colors = plt.rcParams["axes.prop_cycle"]()
fig, axes = plt.subplots(10, 1, sharex=True, sharey=True, figsize=(10, 15))
fig.add_subplot(111, frameon=False)
plt.tick_params(
    labelcolor="none", which="both", top=False, bottom=False, left=False, right=False
)
for k in range(num_classes):
    c = next(colors)["color"]
    axes.flatten()[k].bar(
        np.arange(args["latent_dim"]),
        V_nat[k],
        color=c,
        width=2,
        label="{}".format(classdict.get(k)),
    )
    axes.flatten()[k].legend(loc="upper left", fontsize=15)
    axes.flatten()[k].tick_params(labelsize=15)
plt.xlabel("latent dimensions", size=17)
plt.ylabel("V-nat", size=17)
plt.savefig(
    "{}/all_vnat.png".format(model_path), dpi=100, bbox_inches="tight", pad_inches=0.1
)
plt.show()
plt.close()
#%%
"""Appendix Figure 3"""
df_vnat = pd.DataFrame(V_nat.T, columns=list(classdict.values()))
corr = df_vnat.corr()

plt.subplots(figsize=(7, 7))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    corr,
    cmap="RdYlBu_r",
    annot=True,
    fmt=".3f",
    annot_kws={"size": 9},
    mask=mask,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    vmin=-1,
    vmax=1,
)
plt.savefig(
    "{}/vnat_corr.png".format(model_path), dpi=100, bbox_inches="tight", pad_inches=0.1
)
plt.show()
plt.close()
#%%
data_dir = r"D:\cifar10_{}".format(5000)
idx = np.arange(100)
x = np.array([np.load(data_dir + "/x_{}.npy".format(i)) for i in idx])
y = np.array([np.load(data_dir + "/y_{}.npy".format(i)) for i in idx])
x = (tf.cast(x, tf.float32) - 127.5) / 127.5

_, _, _, _, _, z, images = model(x, training=False)
#%%
"""Train dataset reconstruction: Appendix Figure 6"""
plt.figure(figsize=(15, 15))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow((images[i] + 1) / 2)
    plt.axis("off")
plt.tight_layout()
plt.savefig("{}/train_recon.png".format(model_path))
plt.show()
plt.close()
#%%
"""Test dataset reconstruction: Appendix Figure 7"""
x = x_test[:100]

_, _, _, _, _, _, images = model(x, training=False)
plt.figure(figsize=(15, 15))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow((images[i] + 1) / 2)
    plt.axis("off")
plt.tight_layout()
plt.savefig("{}/test_recon.png".format(model_path))
plt.show()
plt.close()
#%%
"""noise perturbation: Figure 5(b)"""
x = np.load(data_dir + "/x_{}.npy".format(7))
x = (tf.cast(x, tf.float32) - 127.5) / 127.5
mean, logvar, prob, _, _, latent, images = model(x[tf.newaxis, ...], training=False)
# mean = mean[0, np.argmax(prob), :]
latent = np.squeeze(latent)

from copy import deepcopy

noise_z = deepcopy(latent)
keep_z = deepcopy(latent)

tf.random.set_seed(520)
zeros = np.zeros((args["latent_dim"],))

signal_noise = np.zeros((args["latent_dim"],))
signal_noise[V_nat[k] > delta] = tf.random.uniform(
    (sum(V_nat[k] > delta),), -2, 2
).numpy()

signal_keep = np.zeros((args["latent_dim"],))
signal_keep[V_nat[k] <= delta] = tf.random.uniform(
    (sum(V_nat[k] <= delta),), -2, 2
).numpy()

noise_z += signal_noise
keep_z += signal_keep

signals = [zeros, signal_noise, signal_keep]
perturbed_z = [latent, noise_z, keep_z]

clear_img = model.decode(perturbed_z[0][None, ...], training=False)[-1].numpy()
noise_img = model.decode(perturbed_z[1][None, ...], training=False)[-1].numpy()
keep_img = model.decode(perturbed_z[2][None, ...], training=False)[-1].numpy()

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
axes.flatten()[0].plot(signals[0])
axes.flatten()[0].tick_params(labelsize=25)
axes.flatten()[0].set_xlabel("latent dimension", fontsize=24)
axes.flatten()[0].set_ylabel("noise", fontsize=25)
axes.flatten()[0].locator_params(axis="y", nbins=3)
axes.flatten()[1].imshow((clear_img + 1.0) / 2.0)
axes.flatten()[1].axis("off")

axes.flatten()[2].bar(np.arange(args["latent_dim"]), signals[1], width=2)
axes.flatten()[2].tick_params(labelsize=25)
axes.flatten()[2].set_xlabel("latent dimension", fontsize=24)
axes.flatten()[2].locator_params(axis="y", nbins=3)
axes.flatten()[3].imshow((noise_img + 1.0) / 2.0)
axes.flatten()[3].axis("off")

axes.flatten()[4].bar(np.arange(args["latent_dim"]), signals[2], width=2)
axes.flatten()[4].tick_params(labelsize=25)
axes.flatten()[4].set_xlabel("latent dimension", fontsize=24)
axes.flatten()[4].locator_params(axis="y", nbins=3)
axes.flatten()[5].imshow((keep_img + 1.0) / 2.0)
axes.flatten()[5].axis("off")

plt.savefig(
    "{}/blur.png".format(model_path), dpi=100, bbox_inches="tight", pad_inches=0.1
)
plt.show()
plt.close()
#%%
"""Appendix Figure 5"""
x = np.load(data_dir + "/x_{}.npy".format(119))
x = (tf.cast(x, tf.float32) - 127.5) / 127.5
mean, logvar, prob, _, _, z, images = model(x[tf.newaxis, ...], training=False)
# mean = mean[0, np.argmax(prob), :]

clear_rand = deepcopy(z)
clear_rand = np.tile(clear_rand, (21, 1))

tf.random.set_seed(520)
clear_rand[:, V_nat[k] > delta] += tf.random.uniform(
    (21, sum(V_nat[k] > delta)), -1.5, 1.5
).numpy()
clear_rand[0, :] = z
clear_rand_recon = model.decode(clear_rand, training=False)
fig, axes = plt.subplots(3, 7, figsize=(10, 4))
for i in range(21):
    axes.flatten()[i].imshow((clear_rand_recon[i].numpy() + 1.0) / 2.0)
    axes.flatten()[i].axis("off")
plt.savefig(
    "{}/blur_many.png".format(model_path), dpi=100, bbox_inches="tight", pad_inches=0.1
)
plt.show()
plt.close()
#%%
"""single axis perturbation: Appendix Figure 4"""
x = np.load(data_dir + "/x_{}.npy".format(8))
x = (tf.cast(x, tf.float32) - 127.5) / 127.5
mean, logvar, prob, _, _, z, images = model(x[tf.newaxis, ...], training=False)
# mean = mean[0, np.argmax(prob), :]
k = np.argmax(prob)

sorted_idx = np.argsort(V_nat[k])[::-1][: sum(V_nat[k] > delta)]

fig, axes = plt.subplots(5, 11, figsize=(10, 4))
for j in range(5):
    one_rands = np.tile(z, (11, 1))
    one_rands[:, sorted_idx[j]] = 0
    one_rands[:, sorted_idx[j]] += np.round(np.linspace(-3, 3, 11), 2)
    one_rand_recon = model.decode(one_rands, training=False)
    for i in range(11):
        axes[j][i].imshow((one_rand_recon[i].numpy() + 1.0) / 2.0)
        axes[j][i].axis("off")
plt.savefig(
    "{}/oneaxis.png".format(model_path), dpi=100, bbox_inches="tight", pad_inches=0.1
)
plt.show()
plt.close()
#%%
data_dir = r"D:\cifar10_{}".format(5000)
idx = np.arange(100)
x = np.array([np.load(data_dir + "/x_{}.npy".format(i)) for i in idx])
y = np.array([np.load(data_dir + "/y_{}.npy".format(i)) for i in idx])
x = (tf.cast(x, tf.float32) - 127.5) / 127.5

_, _, _, _, _, z, images = model(x, training=False)
#%%
"""interpolation same class: Figure 4"""
fig, axes = plt.subplots(4, 10, figsize=(30, 12))
for idx, (class_idx, i, j) in enumerate([[0, 4, 5], [1, 0, 5], [7, 0, 2], [8, 0, 2]]):
    interpolation_idx = np.where(np.argmax(y, axis=-1) == class_idx)[0]

    inter = np.linspace(z[interpolation_idx[i]], z[interpolation_idx[j]], 8)
    inter_recon = model.decode(inter, training=False)

    axes.flatten()[idx * 10 + 0].imshow((x[interpolation_idx[i]] + 1.0) / 2.0)
    axes.flatten()[idx * 10 + 0].axis("off")
    for i in range(8):
        axes.flatten()[idx * 10 + i + 1].imshow((inter_recon[i].numpy() + 1.0) / 2.0)
        axes.flatten()[idx * 10 + i + 1].axis("off")
    axes.flatten()[idx * 10 + 9].imshow((x[interpolation_idx[j]] + 1.0) / 2.0)
    axes.flatten()[idx * 10 + 9].axis("off")
plt.savefig(
    "{}/interpolation1.png".format(model_path),
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.tight_layout()
plt.show()
plt.close()
#%%
"""interpolation different class: Figure 4"""
fig, axes = plt.subplots(4, 10, figsize=(30, 12))
for idx, (class_idx1, class_idx2, i, j) in enumerate([[0, 8, 5, 0], [1, 7, 0, 0], [1, 0, 6, 4], [7, 8, 1, 2]]):
    interpolation_idx1 = np.where(np.argmax(y, axis=-1) == class_idx1)[0]
    interpolation_idx2 = np.where(np.argmax(y, axis=-1) == class_idx2)[0]

    inter = np.linspace(z[interpolation_idx1[i]], z[interpolation_idx2[j]], 8)
    inter_recon = model.decode(inter, training=False)

    axes.flatten()[idx * 10 + 0].imshow((x[interpolation_idx1[i]] + 1.0) / 2.0)
    axes.flatten()[idx * 10 + 0].axis("off")
    for i in range(8):
        axes.flatten()[idx * 10 + i + 1].imshow((inter_recon[i].numpy() + 1.0) / 2.0)
        axes.flatten()[idx * 10 + i + 1].axis("off")
    axes.flatten()[idx * 10 + 9].imshow((x[interpolation_idx2[j]] + 1.0) / 2.0)
    axes.flatten()[idx * 10 + 9].axis("off")
plt.savefig(
    "{}/interpolation2.png".format(model_path),
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.tight_layout()
plt.show()
plt.close()
#%%
"""inception score: Table 1 and 2"""
# alreay image = [-1, 1] (tanh activation)
def calculate_inception_score(images, n_split=50, eps=1e-16):
    inception = K.applications.InceptionV3(include_top=True)
    scores = list()
    n_part = int(np.floor(images.shape[0] / n_split))
    for i in tqdm.tqdm(range(n_split)):
        ix_start, ix_end = i * n_part, (i + 1) * n_part
        subset = images[ix_start:ix_end]
        subset = subset.astype("float32")
        subset = tf.image.resize(subset, (299, 299), "nearest")
        # predict p(y|x)
        p_yx = inception.predict(subset)
        # calculate p(y)
        p_y = tf.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std
#%%
# 10,000 generated images from sampled latent variables
np.random.seed(1)
generated_images = []
for i in tqdm.tqdm(range(num_classes)):
    for _ in range(10):
        latents = np.random.multivariate_normal(
            prior_means.numpy()[0][i], np.diag(sigma_vector.numpy()[0]), size=(100,)
        )
        images = model.decode(latents, training=False)
        generated_images.extend(images)
generated_images = np.array(generated_images)
np.random.shuffle(generated_images)

# calculate inception score
is_avg, is_std = calculate_inception_score(generated_images)
#%%
with open("{}/result.txt".format(model_path), "w") as file:
    file.write(
        "TEST classification error: {:.2f}%\n\n".format(
            error_count / total_length * 100
        )
    )
    file.write(
        "cardinality of activated latent subspace: {}\n\n".format(sum(V_nat[k] > delta))
    )
    file.write("inception score | mean: {:.2f}, std: {:.2f}\n\n".format(is_avg, is_std))
#%%
'''path w.r.t. beta: Table 1'''
betas = [0.01, 0.05, 0.1, 0.5, 1]
error = {}
cardinality = {}
inception = {}
for b in betas:
    model_path = log_path + '/beta_{}'.format(b)
    with open('{}/result.txt'.format(model_path), 'r') as f:
        line = f.readlines()
    error[b] = line[0].split(' ')[-1][:-2]
    cardinality[b] = line[2].split(' ')[-1][:-1]
    inception[b] = line[4].split(' | ')[-1][:-1]

pd.concat([
    pd.DataFrame.from_dict(inception, orient='index').rename(columns={0: 'Inception score'}).T,
    pd.DataFrame.from_dict(cardinality, orient='index').rename(columns={0: 'cardinality'}).T,
    pd.DataFrame.from_dict(error, orient='index').rename(columns={0: 'test error'}).T,
], axis=0).to_csv(log_path + '/beta_path.csv')
#%%