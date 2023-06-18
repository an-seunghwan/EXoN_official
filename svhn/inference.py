#%%
import argparse
import os

# os.chdir(r'D:\semi\dgm') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/dgm') # main directory (repository)
# os.chdir('/Users/anseunghwan/Documents/GitHub/semi/dgm')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tqdm
import yaml
import io
import matplotlib.pyplot as plt
import random as python_random

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import MixtureVAE
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
    project="EXoN", 
    entity="anseunghwan",
    tags=["svhn", 'EXoN', "inference"],
)
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

    parser.add_argument('--dataset', type=str, default='svhn',
                        help='dataset used for training')
    parser.add_argument('--num', type=int, default=30, # 11 
                        help='seed for repeatable results')
    return parser
#%%
def main():
    #%%
    # '''argparse to dictionary'''
    # args = vars(get_args())
    '''argparse debugging'''
    args = vars(get_args().parse_args(args=[]))
    #%%
    """model load"""
    artifact = wandb.use_artifact(f'anseunghwan/EXoN/{args["dataset"]}_EXoN:v{args["num"]}', type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    model_dir = artifact.download()
    #%%
    np.random.seed(args["seed"])
    python_random.seed(args["seed"])
    tf.random.set_seed(args["seed"])
    #%%
    save_path = os.path.dirname(os.path.abspath(__file__)) + '/dataset/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args['dataset'], save_path, args)
    total_length = sum(1 for _ in datasetU)
    #%%
    model_name = [x for x in os.listdir(model_dir) if x.endswith('h5')][0]
    model = MixtureVAE(
        args, num_classes, latent_dim=args['latent_dim'], dropratio=args['drop_rate']
    )
    model.build(input_shape=(None, 32, 32, 3))
    model.load_weights(model_dir + '/' + model_name)
    model.summary()
    #%%
    # autotune = tf.data.AUTOTUNE
    # shuffle_and_batchL = lambda dataset: dataset.shuffle(
    #     buffer_size=int(1e4),
    #     seed=0).batch(
    #         batch_size=args['labeled_batch_size'], 
    #         drop_remainder=True).prefetch(autotune)
    # iteratorL = iter(shuffle_and_batchL(datasetL))
    # imageL, labelL = next(iteratorL)
    # #%%
    # for i in range(imageL.shape[0]):
    #     np.save(f"./assets/report/img{i}", imageL[i].numpy())
    #     np.save(f"./assets/report/label{i}", labelL[i].numpy())
    #%%
    # target = 4
    # target_list = np.where((labelL == 1).numpy().argmax(axis=1) == target)[0]
    
    z_list = []
    idx_list = [5, 20]
    img_list = []
    for idx in idx_list: # 5, 20
        img = np.load(f"./assets/report/img{idx}.npy")
        label = np.load(f"./assets/report/label{idx}.npy")
        mean, logvar, prob, y, z, z_tilde = model.encode(img[tf.newaxis, ...], training=False)
        z_list.append(z_tilde)
        img_list.append(img)
    
    z_inter = np.linspace(z_list[0][0], z_list[1][0], 8)
    inter_recon = model.decoder(z_inter, training=False)
    
    fig, axes = plt.subplots(1, 10, figsize=(25, 5))
    axes.flatten()[0].imshow((img_list[0] + 1) / 2)
    axes.flatten()[0].axis('off')
    for i in range(8):
        axes.flatten()[i+1].imshow((inter_recon[i].numpy() + 1) / 2)
        axes.flatten()[i+1].axis('off')
    axes.flatten()[9].imshow((img_list[1] + 1) / 2)
    axes.flatten()[9].axis('off')
    plt.tight_layout()
    plt.savefig(f'./assets/exon_svhn_inter_same.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    wandb.log({f'between same': wandb.Image(fig)})
    #%%
    z_list = []
    idx_list = [13, 23]
    img_list = []
    for idx in idx_list:
        img = np.load(f"./assets/report/img{idx}.npy")
        label = np.load(f"./assets/report/label{idx}.npy")
        mean, logvar, prob, y, z, z_tilde = model.encode(img[tf.newaxis, ...], training=False)
        z_list.append(z_tilde)
        img_list.append(img)
    
    z_inter = np.linspace(z_list[0][0], z_list[1][0], 8)
    inter_recon = model.decoder(z_inter, training=False)
    
    fig, axes = plt.subplots(1, 10, figsize=(25, 5))
    axes.flatten()[0].imshow((img_list[0] + 1) / 2)
    axes.flatten()[0].axis('off')
    for i in range(8):
        axes.flatten()[i+1].imshow((inter_recon[i].numpy() + 1) / 2)
        axes.flatten()[i+1].axis('off')
    axes.flatten()[9].imshow((img_list[1] + 1) / 2)
    axes.flatten()[9].axis('off')
    plt.tight_layout()
    plt.savefig(f'./assets/exon_svhn_inter_diff.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    wandb.log({f'between diff': wandb.Image(fig)})
    #%%
    wandb.config.update(args, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%