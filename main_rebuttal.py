#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import scipy.io
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

from modules.mixup import (
    augment,
    
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
    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled_batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--image_size', type=int, default=224, 
                        help='the size of image')
    parser.add_argument('--class_num', type=int, default=20, 
                        help='the number of class')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--labeled_ratio', type=float, default=0.1, 
                        help='ratio of labeled examples (default: 0.1)')
    parser.add_argument('--epochs', default=600, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--reconstruct_freq', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--augment', default=True, type=bool,
                        help="apply augmentation to image")
    parser.add_argument('--aug_pseudo', default=True, type=bool,
                        help="apply augmentation in pseudo label computation")
    parser.add_argument('--dropout_pseudo', default=False, type=bool,
                        help="apply dropout in pseudo label computation")

    '''Deep VAE Model Parameters'''
    parser.add_argument('--drop_rate', default=0.1, type=float, 
                        help='drop rate for the network')

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
    parser.add_argument('--beta', default=0.01, type=float, 
                        help='value of observation noise')
    parser.add_argument('--rampup_epoch', default=50, type=int, 
                        help='the max epoch to adjust unsupervised weight')
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument("--adjust_lr", default=[250], type=arg_as_list, # classifier optimizer scheduling
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="beta distribution parameter")

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * torch.exp(torch.tensor(-5. * (1. - min(1., epoch/epochs)) ** 2))

def train(model,
          buffer_model,
          optimizer,
          optimizer_classifier,
          labeled_dataset,
          unlabeled_dataset, 
          prior_means,
          sigma_vector,
          epoch,
          config,
          device):
    
    logs = {
        "loss": [],
        "elbo": [],
        "recon": [],
        "kl": [],
        "cce": [],
        "mixup_yL": [],
        "mixup_yU": [],
    }
    
    lambda2 = weight_schedule(epoch, config['rampup_epoch'], config["lambda1"])

    iteration = unlabeled_dataset.__len__() // config["batch_size"]
    for _ in tqdm.tqdm(range(iteration), desc='inner loop'):
        try:
            imageL, labelL = next(iteratorL)
        except:
            labeled_dataloader = DataLoader(labeled_dataset, batch_size=config["labeled_batch_size"], shuffle=True)
            iteratorL = iter(labeled_dataloader)
            imageL, labelL = next(iteratorL)
        try:
            imageU = next(iteratorU)
        except:
            unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=config["batch_size"], shuffle=True)
            iteratorU = iter(unlabeled_dataloader)
            imageU = next(iteratorU)

        if config["cuda"]:
            imageL = imageL.to(device)
            labelL = labelL.to(device)
            imageU = imageU.to(device)

        if config['augment']:
            imageL_aug = augment(imageL, config)
            imageU_aug = augment(imageU, config)
        
        if config["cuda"]:
            imageL_aug = imageL_aug.to(device)
            imageU_aug = imageU_aug.to(device)

        # non-augmented image
        image = torch.cat([imageL, imageU], dim=0) 

        '''mix-up weight'''
        mix_weight = [np.random.beta(config['epsilon'], config['epsilon']), # labeled
                      np.random.beta(2.0, 2.0)] # unlabeled

        optimizer.zero_grad()
        optimizer_classifier.zero_grad()

        loss_ = []

        '''ELBO'''
        mean, logvar, probs, y, z, z_tilde, xhat = model(image)

        error = (0.5 * torch.pow(image - xhat, 2).sum(axis=[1, 2, 3])).sum()

        kl1 = torch.log(probs) + torch.log(torch.tensor(config["class_num"]))
        kl1 = (probs * kl1).sum(axis=1).sum()

        kl2 = torch.pow(mean - prior_means, 2) / sigma_vector
        kl2 -= 1
        kl2 += torch.log(sigma_vector)
        kl2 += torch.exp(logvar) / sigma_vector
        kl2 -= logvar
        kl2 = probs * (0.5 * kl2).sum(axis=-1)
        kl2 = kl2.sum()

        probL_aug = model.classify(imageL_aug)
        cce = F.nll_loss(torch.log(probL_aug), labelL.squeeze().type(torch.long),
                        reduction='none').sum()

        '''soft-label consistency interpolation'''
        # mix-up
        with torch.no_grad():
            indices = torch.randperm(imageL_aug.size(0)).to(device)
            image_shuffleL = torch.index_select(imageL_aug, dim=0, index=indices)
            label_shuffleL = torch.index_select(labelL, dim=0, index=indices)
            image_mixL = mix_weight[0] * image_shuffleL + (1 - mix_weight[0]) * imageL_aug
            
            if config['aug_pseudo']:
                pseudo_labelU = buffer_model.classify(imageU_aug)
            else:
                pseudo_labelU = buffer_model.classify(imageU)
            indices = torch.randperm(imageU_aug.size(0)).to(device)
            image_shuffleU = torch.index_select(imageU_aug, dim=0, index=indices)
            pseudo_label_shuffleU = torch.index_select(pseudo_labelU, dim=0, index=indices)
            image_mixU = mix_weight[1] * image_shuffleU + (1 - mix_weight[1]) * imageU_aug
            
        # labeled
        prob_mixL = model.classify(image_mixL)
        mixup_yL = F.nll_loss(torch.log(prob_mixL), label_shuffleL.squeeze().type(torch.long),
                            reduction='none').sum() * mix_weight[0]
        mixup_yL += F.nll_loss(torch.log(prob_mixL), labelL.squeeze().type(torch.long),
                            reduction='none').sum() * (1 - mix_weight[0])
        # unlabeled
        prob_mixU = model.classify(image_mixU)
        mixup_yU = - (pseudo_label_shuffleU * torch.log(prob_mixU)).sum(axis=1).sum() * mix_weight[1]
        mixup_yU += - (pseudo_labelU * torch.log(prob_mixU)).sum(axis=1).sum() * (1 - mix_weight[1])

        elbo = error + config["beta"] * (kl1 + kl2 + cce)
        loss = elbo + config["lambda1"] * (cce + mixup_yL) + lambda2 * mixup_yU
        
        loss_.append(('loss', loss))
        loss_.append(('elbo', elbo))
        loss_.append(('recon', error))
        loss_.append(('kl', kl1 + kl2))
        loss_.append(('cce', cce))
        loss_.append(('mixup_yL', mixup_yL))
        loss_.append(('mixup_yU', mixup_yU))

        # encoder and decoder
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer_classifier.step()

        ""'weight decay'""
        for param, buffer_param in zip(model.classifier.parameters(), buffer_model.classifier.parameters()):
            decay_rate = config['weight_decay'] * optimizer_classifier.param_groups[0]['lr']
            param.data = param - decay_rate * buffer_param
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, xhat
#%%
def main():
    #%%
    """configuration"""
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)

    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """dataset"""
    cars_annos = scipy.io.loadmat('./standford_car/cars_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    train_imgs = []
    test_imgs = []
    train_labels = []
    test_labels = []
    for anno in tqdm.tqdm(annotations):
        if anno[0][-1][0][0] == 0: # train
            if anno[0][-2][0][0] <= config["class_num"]:
                train_labels.append(anno[0][-2][0][0] - 1)
                train_imgs.append(anno[0][0][0])
        else: # test
            if anno[0][-2][0][0] <= config["class_num"]:
                test_labels.append(anno[0][-2][0][0] - 1)
                test_imgs.append(anno[0][0][0])
    
    # label and unlabel index
    idx = np.random.choice(range(len(train_imgs)), 
                            int(len(train_imgs) * config["labeled_ratio"]), 
                            replace=False)

    labeled_dataset = LabeledDataset(train_imgs, train_labels, config, idx)
    unlabeled_dataset = UnLabeledDataset(train_imgs, config)
    test_dataset = TestDataset(test_imgs, test_labels, config, idx)

    # test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    #%%
    """model"""
    model = MixtureVAE(
        config, class_num=config["class_num"], dropratio=config['drop_rate'], device=device
        ).to(device)

    if config['dropout_pseudo']:
        buffer_model = MixtureVAE(
            config, class_num=config["class_num"], dropratio=config['drop_rate'], device=device
        ).to(device)
    else:
        buffer_model = MixtureVAE(
            config, class_num=config["class_num"], dropratio=0, device=device
        ).to(device)
    # initialize weights
    for param, buffer_param in zip(model.parameters(), buffer_model.parameters()):
        buffer_param.data = param
    #%%
    """optimizer"""
    optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()), 
            lr=config["lr"]
        )
    optimizer_classifier = torch.optim.Adam(
            model.classifier.parameters(), 
            lr=config["lr"]
        )
        
    model.train()
    #%%
    '''prior design'''
    prior_means = np.zeros((config["class_num"], config['latent_dim']))
    prior_means[:, :config["class_num"]] = np.eye(config["class_num"]) * config['dist']
    prior_means = torch.tensor(prior_means[np.newaxis, :, :], dtype=torch.float32).to(device)

    sigma_vector = np.ones((1, config['latent_dim'])) 
    sigma_vector[0, :config["class_num"]] = config['sigma1']
    sigma_vector[0, config["class_num"]:] = config['sigma2']
    sigma_vector = torch.tensor(sigma_vector, dtype=torch.float32).to(device)
    #%%
    for epoch in range(config["epochs"]):
        if epoch == 0:
            """warm-up"""
            for g in optimizer.param_groups:
                g['lr'] = config["lr"] * 0.1

        # """classifier: learning rate schedule"""
        if epoch == 0:
            """warm-up"""
            for g in optimizer_classifier.param_groups:
                g['lr'] = config["lr"] * 0.2
        # else:
        #     """exponential decay"""
        #     if epoch >= config["adjust_lr"][-1]:
        #         lr_ = config['lr'] * (config['lr_gamma'] ** len(config['adjust_lr']))
        #         for g in optimizer_classifier.param_groups:
        #             g['lr'] = lr_ * torch.exp(-5. * (1. - (config['epochs'] - epoch) / (config['epochs'] - config['adjust_lr'][-1])) ** 2)
        #     else:
        #         '''constant decay'''
        #         for ad_num, ad_epoch in enumerate(config['adjust_lr']): 
        #             if epoch < ad_epoch:
        #                 for g in optimizer_classifier.param_groups:
        #                     g['lr'] = config['lr'] * (config['lr_gamma'] ** ad_num)
                    
        
        logs, xhat = train(model,
                        buffer_model,
                        optimizer,
                        optimizer_classifier,
                        labeled_dataset,
                        unlabeled_dataset, 
                        prior_means,
                        sigma_vector,
                        epoch,
                        config,
                        device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
            
        if epoch % config["reconstruct_freq"] == 0:
            plt.figure(figsize=(4, 4))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow((np.transpose(xhat[i].cpu().detach().numpy(), [1, 2, 0]) + 1) / 2)
                plt.axis('off')
            plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
            plt.close()
        
        if epoch == 0:
            for g in optimizer.param_groups:
                g['lr'] = config["lr"]
            for g in optimizer_classifier.param_groups:
                g['lr'] = config["lr"]
    #%%
    """reconstruction result"""
    fig = plt.figure(figsize=(4, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow((np.transpose(xhat[i].cpu().detach().numpy(), [1, 2, 0]) + 1) / 2)
        plt.axis('off')
    plt.savefig('./assets/recon.png')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})
    #%%
    """model save"""
    torch.save(model.state_dict(), './assets/model.pth')
    artifact = wandb.Artifact('model', 
                              type='model',
                              metadata=config) # description=""
    artifact.add_file('./assets/model.pth')
    artifact.add_file('./main_rebuttal.py')
    artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%