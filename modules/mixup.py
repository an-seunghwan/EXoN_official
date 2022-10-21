#%%
import torchvision.transforms.functional as TF
import random
from torchvision.transforms import *
#%%
def augment(image, config):
    if random.random() > 0.5:
        image = TF.vflip(image)
    image = TF.pad(image, padding=[2, 2], padding_mode='reflect')
    image = RandomCrop(size=config["image_size"])(image)
    return image
#%%