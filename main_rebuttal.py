#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.model import (
    MixtureVAE
)

from modules.standford_car import (
    LabeledDataset, 
    UnLabeledDataset,
    TestDataset
)
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="EXoN(Rebuttal)", 
    entity="anseunghwan",
    tags=["Standford_Cars"],
)
#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='standford_car')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled_batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--image_size', type=int, default=224, 
                        help='the size of image')
    parser.add_argument('--class_num', type=int, default=196, 
                        help='the number of class')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--reconstruct_freq', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_ratio', type=float, default=0.1, 
                        help='ratio of labeled examples (default: 0.1)')
    parser.add_argument('--augment', default=True, type=bool,
                        help="apply augmentation to image")
    parser.add_argument('--aug_pseudo', default=True, type=bool,
                        help="apply augmentation in pseudo label computation")
    parser.add_argument('--dropout_pseudo', default=False, type=bool,
                        help="apply dropout in pseudo label computation")

    '''Deep VAE Model Parameters'''
    parser.add_argument('--drop_rate', default=0.1, type=float, 
                        help='drop rate for the network')
    parser.add_argument("--bce_reconstruction", default=False, type=bool,
                        help="Do BCE Reconstruction")

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

    '''VAE Loss Function Parameters'''
    parser.add_argument('--lambda1', default=5000, type=int, 
                        help='the weight of classification loss term')
    parser.add_argument('--beta', default=0.01, type=int, 
                        help='value of observation noise')
    parser.add_argument('--rampup_epoch', default=50, type=int, 
                        help='the max epoch to adjust unsupervised weight')
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument("--adjust_lr", default=[250, 350, 450, 550], type=arg_as_list, # classifier optimizer scheduling
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="beta distribution parameter")

    '''Configuration'''
    parser.add_argument('--config_path', type=str, default=None, 
                        help='path to yaml config file, overwrites args')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
config = vars(get_args(debug=True)) # default configuration
config["cuda"] = torch.cuda.is_available()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
wandb.config.update(config)

torch.manual_seed(config["seed"])
if config["cuda"]:
    torch.cuda.manual_seed(config["seed"])
#%%
"""dataset"""
import scipy.io
cars_annos = scipy.io.loadmat('./standford_car/cars_annos.mat')
annotations = cars_annos['annotations']
annotations = np.transpose(annotations)

train_imgs = []
test_imgs = []
train_labels = []
test_labels = []
for anno in tqdm.tqdm(annotations):
    if anno[0][-1][0][0] == 0: # train
        train_labels.append(anno[0][-2][0][0])
        train_imgs.append(anno[0][0][0])
    else: # test
        test_labels.append(anno[0][-2][0][0])
        test_imgs.append(anno[0][0][0])
        
# label and unlabel index
idx = np.random.choice(range(len(train_imgs)), 
                        int(len(train_imgs) * config["labeled_ratio"]), 
                        replace=False)

labeled_dataset = LabeledDataset(train_imgs, train_labels, config, idx)
unlabeled_dataset = UnLabeledDataset(train_imgs, config)
test_dataset = TestDataset(test_imgs, test_labels, config, idx)

test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
#%%
"""model"""
model = MixtureVAE(config, class_num=config["class_num"], dropratio=config['drop_rate'])
model = model.to(device)

if config['dropout_pseudo']:
    buffer_model = MixtureVAE(
        config, class_num=config["class_num"], dropratio=config['drop_rate']
    )
else:
    buffer_model = MixtureVAE(
        config, class_num=config["class_num"], dropratio=0
    )
buffer_model = buffer_model.to(device)
# initialize weights
for param, buffer_param in zip(model.parameters(), buffer_model.parameters()):
    buffer_param.data = param
#%%
optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
optimizer_classifier = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
model.train()
#%%
'''prior design'''
prior_means = np.zeros((config["class_num"], config['latent_dim']))
prior_means[:, :config["class_num"]] = np.eye(config["class_num"]) * config['dist']
prior_means = torch.tensor(prior_means[np.newaxis, :, :], dtype=torch.float32)

sigma_vector = np.ones((1, config['latent_dim'])) 
sigma_vector[0, :config["class_num"]] = config['sigma1']
sigma_vector[0, config["class_num"]:] = config['sigma2']
sigma_vector = torch.tensor(sigma_vector, dtype=torch.float32)
#%%
# for epoch in range(config["epochs"]):
epoch = 0
if epoch == 0:
    """warm-up"""
    for g in optimizer.param_groups:
        g['lr'] = config["lr"] * 0.1

"""classifier: learning rate schedule"""
if epoch == 0:
    """warm-up"""
    for g in optimizer_classifier.param_groups:
        g['lr'] = config["lr"] * 0.2
else:
    """exponential decay"""
    if epoch >= config["adjust_lr"][-1]:
        lr_ = config['lr'] * (config['lr_gamma'] ** len(config['adjust_lr']))
        for g in optimizer_classifier.param_groups:
            g['lr'] = lr_ * torch.exp(-5. * (1. - (config['epochs'] - epoch) / (config['epochs'] - config['adjust_lr'][-1])) ** 2)
    else:
        '''constant decay'''
        for ad_num, ad_epoch in enumerate(config['adjust_lr']): 
            if epoch < ad_epoch:
                for g in optimizer_classifier.param_groups:
                    g['lr'] = config['lr'] * (config['lr_gamma'] ** ad_num)
                break
#%%
"""train"""
'''un-supervised reconstruction weight'''
lambda2 = weight_schedule(epoch, config['rampup_epoch'], config["lambda1"])

labeled_dataloader = DataLoader(labeled_dataset, batch_size=config["labeled_batch_size"], shuffle=True)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=config["batch_size"], shuffle=True)

iteratorL = iter(labeled_dataloader)
iteratorU = iter(unlabeled_dataloader)
    
iteration = unlabeled_dataset.__len__() // config["batch_size"]

progress_bar = tqdm.tqdm(range(iteration), unit='batch')
for batch_num in progress_bar:
    
    try:
        imageL, labelL = next(iteratorL)
    except:
        labeled_dataloader = DataLoader(labeled_dataset, batch_size=config["labeled_batch_size"], shuffle=True)
        iteratorL = iter(labeled_dataloader)
        imageL, labelL = next(iteratorL)
    try:
        imageU, _ = next(iteratorU)
    except:
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=config["batch_size"], shuffle=True)
        iteratorU = iter(unlabeled_dataloader)
        imageU, _ = next(iteratorU)
    
    if config['augment']:
        imageL_aug = augment(imageL)
        imageU_aug = augment(imageU)
    
    # non-augmented image
    image = tf.concat([imageL, imageU], axis=0) 
    
    '''mix-up weight'''
    mix_weight = [tf.constant(np.random.beta(args['epsilon'], args['epsilon'])), # labeled
                tf.constant(np.random.beta(2.0, 2.0))] # unlabeled
    
    with tf.GradientTape(persistent=True) as tape:    
        '''ELBO'''
        mean, logvar, prob, y, z, z_tilde, xhat = model(image)
        recon_loss, kl1, kl2 = ELBO_criterion(
            prob, xhat, image, mean, logvar, prior_means, sigma_vector, num_classes, args
        )
        probL_aug = model.classify(imageL_aug)
        cce = - tf.reduce_sum(tf.reduce_sum(tf.multiply(labelL, tf.math.log(tf.clip_by_value(probL_aug, 1e-10, 1.))), axis=-1))
        
        '''soft-label consistency interpolation'''
        # mix-up
        with tape.stop_recording():
            image_mixL, label_shuffleL = non_smooth_mixup(imageL_aug, labelL, mix_weight[0])
            if args['aug_pseudo']:
                pseudo_labelU = buffer_model.classify(imageU_aug)
            else:
                pseudo_labelU = buffer_model.classify(imageU)
            image_mixU, pseudo_label_shuffleU = non_smooth_mixup(imageU_aug, pseudo_labelU, mix_weight[1])
        # labeled
        prob_mixL = model.classify(image_mixL)
        mixup_yL = - tf.reduce_sum(mix_weight[0] * tf.reduce_sum(label_shuffleL * tf.math.log(tf.clip_by_value(prob_mixL, 1e-10, 1.0)), axis=-1))
        mixup_yL += - tf.reduce_sum((1. - mix_weight[0]) * tf.reduce_sum(labelL * tf.math.log(tf.clip_by_value(prob_mixL, 1e-10, 1.0)), axis=-1))
        # unlabeled
        prob_mixU = model.classify(image_mixU)
        mixup_yU = - tf.reduce_sum(mix_weight[1] * tf.reduce_sum(pseudo_label_shuffleU * tf.math.log(tf.clip_by_value(prob_mixU, 1e-10, 1.0)), axis=-1))
        mixup_yU += - tf.reduce_sum((1. - mix_weight[1]) * tf.reduce_sum(pseudo_labelU * tf.math.log(tf.clip_by_value(prob_mixU, 1e-10, 1.0)), axis=-1))
        
        elbo = recon_loss + beta * (kl1 + kl2 + cce)

        loss = elbo + lambda1 * (cce + mixup_yL) + lambda2 * mixup_yU
    
    # encoder and decoder
    grads = tape.gradient(loss, model.decoder.trainable_variables + model.encoder.trainable_variables) 
    optimizer.apply_gradients(zip(grads, model.decoder.trainable_variables + model.encoder.trainable_variables)) 
    # classifier
    grads = tape.gradient(loss, model.classifier.trainable_variables) 
    optimizer_classifier.apply_gradients(zip(grads, model.classifier.trainable_variables)) 
    '''decoupled weight decay'''
    weight_decay_decoupled(model.classifier, buffer_model.classifier, decay_rate=args['weight_decay'] * optimizer_classifier.lr)
    
    loss_avg(loss)
    recon_loss_avg(recon_loss / args['batch_size'])
    kl1_loss_avg(kl1 / args['batch_size'])
    kl2_loss_avg(kl2 / args['batch_size'])
    label_mixup_loss_avg(mixup_yL / args['batch_size'])
    unlabel_mixup_loss_avg(mixup_yU / args['batch_size'])
    probL = model.classify(imageL, training=False)
    accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)
    
    progress_bar.set_postfix({
        'EPOCH': f'{epoch:04d}',
        'Loss': f'{loss_avg.result():.4f}',
        'Recon': f'{recon_loss_avg.result():.4f}',
        'KL1': f'{kl1_loss_avg.result():.4f}',
        'KL2': f'{kl2_loss_avg.result():.4f}',
        'MixUp(L)': f'{label_mixup_loss_avg.result():.4f}',
        'MixUp(U)': f'{unlabel_mixup_loss_avg.result():.4f}',
        'Accuracy': f'{accuracy.result():.3%}',
        'Test Accuracy': f'{test_accuracy_print:.3%}',
        'beta': f'{beta:.4f}'
    })
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * torch.exp(torch.tensor(-5. * (1. - min(1., epoch/epochs)) ** 2))
#%%