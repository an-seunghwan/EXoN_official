#%%
"""
Reference:
[1]: https://github.com/mazenmel/Stanford-Car-Dataset/blob/master/stanfordCars.ipynb
[2]: https://ieeexplore.ieee.org/document/6755945
"""
#%%
import numpy as np
from PIL import Image
import tqdm

import torch
from torch.utils.data import Dataset

# os.chdir('/Users/anseunghwan/Documents/GitHub/EXoN/src')
#%%
# config = {
#     "image_size": 224,
#     "labeled_ratio": 0.1,
#     "batch_size": 4,
# }
#%%
# import scipy.io
# cars_annos = scipy.io.loadmat('/Users/anseunghwan/Documents/GitHub/EXoN/src/standford_car/cars_annos.mat')
# annotations = cars_annos['annotations']
# annotations = np.transpose(annotations)

# train_imgs = []
# test_imgs = []
# train_labels = []
# test_labels = []
# for anno in tqdm.tqdm(annotations):
#     if anno[0][-1][0][0] == 0: # train
#         train_labels.append(anno[0][-2][0][0])
#         train_imgs.append(anno[0][0][0])
#     else: # test
#         test_labels.append(anno[0][-2][0][0])
#         test_imgs.append(anno[0][0][0])
        
# # label and unlabel index
# idx = np.random.choice(range(len(train_imgs)), 
#                         int(len(train_imgs) * config["labeled_ratio"]), 
#                         replace=False)
#%%
class LabeledDataset(Dataset): 
    def __init__(self, train_imgs, train_labels, image_size, idx):
        train_imgs = [train_imgs[i] for i in idx]
        
        train_x = []
        for i in tqdm.tqdm(range(len(train_imgs)), desc="labeled train data loading"):
            train_x.append(
                np.moveaxis(np.array(
                    Image.open("./standford_car/{}".format(train_imgs[i])).resize(
                        (image_size, image_size)).convert('RGB')), -1, 0))
        self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
        
        train_labels = np.array(train_labels).astype(float)[:, None]
        train_labels = train_labels[idx]
        self.y_data = train_labels

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y 
#%%
class UnLabeledDataset(Dataset): 
    def __init__(self, train_imgs, image_size):
        train_x = []
        for i in tqdm.tqdm(range(len(train_imgs)), desc="unlabeled train data loading"):
            train_x.append(
                np.moveaxis(np.array(
                    Image.open("./standford_car/{}".format(train_imgs[i])).resize(
                        (image_size, image_size)).convert('RGB')), -1, 0))
        self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        return x
#%%
class TestDataset(Dataset): 
    def __init__(self, test_imgs, test_labels, image_size):
        test_x = []
        for i in tqdm.tqdm(range(len(test_imgs)), desc="labeled test data loading"):
            test_x.append(
                np.moveaxis(np.array(
                    Image.open("./standford_car/{}".format(test_imgs[i])).resize(
                        (image_size, image_size)).convert('RGB')), -1, 0))
        self.x_data = (np.array(test_x).astype(float) - 127.5) / 127.5
        
        test_labels = np.array(test_labels).astype(float)[:, None]
        self.y_data = test_labels

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y 
#%%
# labeled_dataset = LabeledDataset(train_imgs, train_labels, config, idx)
# unlabeled_dataset = UnLabeledDataset(train_imgs, config, idx)
# test_dataset = TestDataset(test_imgs, test_labels, config, idx)
#%%