#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

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
    tags=["Standford_Cars", "Inference"],
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

    parser.add_argument('--num', type=int, default=8, # 20
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    """configuration"""
    
    class_num = 20
    image_size = 224
    
    config = vars(get_args(debug=True)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    #%%
    """load model"""
    artifact = wandb.use_artifact('anseunghwan/EXoN(Rebuttal)/model:v{}'.format(config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert config["class_num"] == class_num
    assert config["image_size"] == image_size
    wandb.config.update(config)
    
    model_dir = artifact.download()
    model = MixtureVAE(
        config, class_num=config["class_num"], dropratio=config['drop_rate'], device=device
        ).to(device)
    
    if config["cuda"]:
        model.load_state_dict(torch.load(model_dir + '/model.pth'))
    else:
        model.load_state_dict(torch.load(model_dir + '/model.pth', map_location=torch.device('cpu')))
    model.eval()
    #%%
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """dataset"""
    cars_annos = scipy.io.loadmat('./standford_car/cars_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    train_imgs = []
    train_labels = []
    for anno in tqdm.tqdm(annotations):
        if anno[0][-1][0][0] == 0: # train
            if anno[0][-2][0][0] <= class_num:
                train_labels.append(anno[0][-2][0][0] - 1)
                train_imgs.append(anno[0][0][0])
    
    idx = np.random.choice(np.arange(len(train_imgs)), len(train_imgs), replace=False)
    dataset = LabeledDataset(train_imgs, train_labels, image_size, idx)
    #%%
    """reconstruction"""
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False)
    iterator = iter(dataloader)
    count = 4
    for _ in range(count):
        image, label = next(iterator)
        if config["cuda"]:
            image = image.to(device)
            label = label.to(device)
            
    with torch.no_grad():
        mean, logvar, probs, y, z, z_tilde, xhat = model(image, sampling=False)
        label_ = F.one_hot(label.type(torch.int64), num_classes=config["class_num"]).type(torch.float32)
        mean_ = torch.matmul(label_, mean).squeeze(1)
        xhat = model.decode(mean_)
    fig = plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow((np.transpose(xhat[i].cpu().numpy(), [1, 2, 0]) + 1) / 2)
        plt.axis('off')
    # plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
    plt.tight_layout()
    plt.show()
    plt.close()
    
    wandb.log({'reconstruction': wandb.Image(fig)})
    #%%
    """interpolation"""
    idx_pair = (3, 15, 16)
    idx_pair = (7, 5, 18)
    idx_pair = (9, 3, 21)
    idx_pair = (21, 3, 9)
    idx_pair = (22, 12, 21)
    
    fig = plt.figure(figsize=(9, 5))
    
    for k, idx_pair in enumerate([(3, 15, 16), (7, 5, 18), (9, 3, 21), (21, 3, 9), (22, 12, 21)]):
    
        iterator = iter(dataloader)
        count = idx_pair[0]
        for _ in range(count):
            image, label = next(iterator)
            if config["cuda"]:
                image = image.to(device)
                label = label.to(device)
        
        with torch.no_grad():
            mean, logvar, probs, y, z, z_tilde, xhat = model(image, sampling=False)
            label_ = F.one_hot(label.type(torch.int64), num_classes=config["class_num"]).type(torch.float32)
            mean_ = torch.matmul(label_, mean).squeeze(1)
        #     xhat = model.decode(mean_)
        # fig = plt.figure(figsize=(5, 5))
        # for i in range(25):
        #     plt.subplot(5, 5, i+1)
        #     plt.imshow((np.transpose(xhat[i].cpu().numpy(), [1, 2, 0]) + 1) / 2)
        #     plt.axis('off')
        # # plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
        # plt.tight_layout()
        # plt.show()
        # plt.close()
        
        num = 9
        with torch.no_grad():
            mean = mean_.cpu().numpy()
            mean_inter = np.linspace(mean[idx_pair[1]], mean[idx_pair[2]], num)
            xhat_inter = model.decode(torch.from_numpy(mean_inter).to(device))
        
        for i in range(num):
            plt.subplot(5, num, k * num + i + 1)
            plt.imshow((np.transpose(xhat_inter[i].cpu().numpy(), [1, 2, 0]) + 1) / 2)
            plt.axis('off')
    # plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
    plt.tight_layout()
    plt.show()
    plt.close()
    
    wandb.log({'interpolation': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%