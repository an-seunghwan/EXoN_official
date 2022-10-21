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
    parser.add_argument('--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled-batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000), all labels are balanced')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
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
    parser.add_argument('--learning_rate', default=0.001, type=float,
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
import scipy.io
cars_annos = scipy.io.loadmat('/Users/anseunghwan/Documents/GitHub/EXoN/src/standford_car/cars_annos.mat')
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
#%%