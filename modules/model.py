#%%
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#%%
class Encoder(nn.Module):
    def __init__(self, latent_dim, class_num, channel=3, device='cpu'):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.channel_dim = 128
        self.class_num = class_num
        
        self.net = nn.Sequential(
            nn.Conv2d(channel, self.channel_dim, 4, 2, 1),
            nn.BatchNorm2d(self.channel_dim),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.channel_dim, self.channel_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.channel_dim * 2, self.channel_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.channel_dim * 4, self.channel_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_dim * 8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.channel_dim * 8, self.channel_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_dim * 8),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(self.channel_dim * 8 * 7 * 7, 
                      latent_dim * 2 * class_num),
        ).to(device)
        
    def forward(self, input):
        h = self.net(input)
        mean, logvar = torch.split(h, split_size_or_sections=self.latent_dim * self.class_num, dim=-1)
        mean = torch.split(mean, split_size_or_sections=self.latent_dim, dim=-1)
        logvar = torch.split(logvar, split_size_or_sections=self.latent_dim, dim=-1)
        return torch.stack(mean, dim=1), torch.stack(logvar, dim=1)
#%%
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, self.shape[0], self.shape[1], self.shape[2])
#%%
class Decoder(nn.Module):
    def __init__(self, latent_dim, channel=3, device='cpu'):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.channel = channel
        self.channel_dim = 128
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 
                      self.channel_dim * 8 * 7 * 7),
            View((self.channel_dim * 8, 7, 7)),
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.channel_dim * 8, self.channel_dim * 8, 3, 1),
            nn.BatchNorm2d(self.channel_dim * 8),
            nn.LeakyReLU(0.2),
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.channel_dim * 8, self.channel_dim * 4, 3, 1),
            nn.BatchNorm2d(self.channel_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.channel_dim * 4, self.channel_dim * 2, 3, 1),
            nn.BatchNorm2d(self.channel_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.channel_dim * 2, self.channel_dim, 3, 1),
            nn.BatchNorm2d(self.channel_dim),
            nn.LeakyReLU(0.2),
            
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.channel_dim , self.channel, 3, 1),
            nn.Tanh(),
        ).to(device)
        
    def forward(self, input):
        h = self.net(input)
        return h
#%%
class Classifier(nn.Module):
    def __init__(self, class_num, drop_ratio, channel=3, device='cpu'):
        super(Classifier, self).__init__()
        self.class_num = class_num
        self.channel_dim = 128
        p = 0.5
        
        self.net = nn.Sequential(
            nn.Conv2d(channel, self.channel_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.channel_dim, self.channel_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.channel_dim, self.channel_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim),
            nn.LeakyReLU(0.1),
            
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(drop_ratio),
            
            nn.Conv2d(self.channel_dim, self.channel_dim * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim * 2),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.channel_dim * 2, self.channel_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim * 2),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.channel_dim * 2, self.channel_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim * 2),
            nn.LeakyReLU(0.1),
            
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(drop_ratio),
            
            nn.Conv2d(self.channel_dim * 2, self.channel_dim * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim * 4),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.channel_dim * 4, self.channel_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim * 2),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(self.channel_dim * 2, self.channel_dim * 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel_dim * 1),
            nn.LeakyReLU(0.1),
            
            nn.AdaptiveAvgPool2d(1),
        ).to(device)
        
        self.fc = nn.Linear(self.channel_dim, class_num)
        
    def forward(self, input):
        out = self.net(input)
        out = nn.Softmax(dim=1)(self.fc(out.squeeze()))
        return out
#%%
class MixtureVAE(nn.Module):
    def __init__(self, config, class_num, device='cpu'):
        super(MixtureVAE, self).__init__()
        self.config = config
        self.device = device
        
        self.encoder = Encoder(config["latent_dim"], class_num, device=device)
        self.decoder = Decoder(config["latent_dim"], device=device)
        self.classifier = Classifier(class_num, device=device)
        
    def sample_gumbel(self, shape):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + 1e-8) + 1e-8)

    def gumbel_max_sample(self, probs):
        y = torch.log(probs + 1e-8) + self.sample_gumbel(probs.shape)
        if self.config["hard"]:
            y_hard = (y == torch.max(y, 1, keepdim=True)[0]).type(y.dtype)
            y = (y_hard - y).detach() + y
        return y
    
    def encode(self, input):
        mean, logvar = self.encoder(input)
        return mean, logvar
    
    def classify(self, input):
        probs = self.classifier(input)
        return probs
    
    def decode(self, input):
        xhat = self.decoder(input)
        return xhat
    
    def forward(self, input, sampling=True):
        mean, logvar = self.encoder(input)
        
        if sampling:
            epsilon = torch.randn(mean.shape).to(self.device)
            z = mean + epsilon * torch.exp(logvar / 2)
        else:
            z = mean
            
        probs = self.classifier(input)
        y = self.gumbel_max_sample(probs)
        
        z_tilde = torch.matmul(y[:, None, :], z).squeeze(1)
        
        xhat = self.decoder(z_tilde)
        
        return mean, logvar, probs, y, z, z_tilde, xhat
#%%
def main():
    #%%
    config = {
        "image_size": 224,
        "latent_dim": 256,
        "hard": True,
    }
    batch = torch.randn(3, 3, config["image_size"], config["image_size"])
    
    encoder = Encoder(config["latent_dim"], class_num=10)
    mean, logvar = encoder(batch)
    mean.shape
    logvar.shape
    
    decoder = Decoder(config["latent_dim"])
    out = decoder(mean)
    out.shape
    
    classifier = Classifier(class_num=10)
    out = classifier(batch)
    out.shape
    
    model = MixtureVAE(config, class_num=10)
    mean, logvar, probs, y, z, z_tilde, xhat = model(batch)
    mean.shape
    logvar.shape
    probs.shape
    y.shape
    z.shape
    z_tilde.shape
    xhat.shape
#%%
if __name__ == '__main__':
    main()
#%%