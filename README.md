# EXoN: EXplainable encoder Network

This repository is the official implementation of [EXoN: EXplainable encoder Network](https://arxiv.org/abs/2105.10867) with Tensorflow 2.0. 

## Requirements

```setup
python 3.7
tensorflow 2.4.3
```

## Training 

To follow our step-by-step tutorial implementations of our proposed VAE model in the paper, use following codes:

### 0. modules and source codes 

```
.
+-- assets 
    +-- mnist (folder which contains results and trained weights from the MNIST dataset experiments)
    +-- cifar10 (folder which contains results and trained weights from the CIFAR-10 dataset experiments)
+-- src
|   +-- modules
|       +-- __init__.py
|       +-- MNIST.py 
|       +-- CIFAR10.py 
|   +-- exon_mnist.py
|   +-- exon_mnist_path.py
|   +-- exon_mnist_design.py
|   +-- exon_cifar10.py
|   +-- exon_cifar10_path.py
|   +-- exon_cifar10_result.py
|   +-- utils
|       +-- exon_mnist_classification.py 
|       +-- exon_cifar10_classification.py 
|       +-- exon_cifar10_datasave.py (save each CIFAR-10 image into numpy array)
+-- README.md
+-- LICENSE
```

### 1. MNIST Dataset

Manual parameter setting:
```
batch_size: mini-batch size of the unlabeled dataset
data_dim: dimension size of an observation (image, flattened)
class_num: the number of labels
latent_dim: dimension size of latent variable
sigma: diagonal variance element of prior distribution
activation: activation function of output layer of decoder
iterations: the number of total iterations
lambda1: tuning parameter of classification error
lambda2: tuning parameter of 1/beta regularization
learning_rate: learning rate of optimizer
labeled: the number of labeled observations
hard: If True, Gumbel-Max trick is used at forward pass
FashionMNIST: If True, use Fashion MNIST dataset instead of MNIST dataset
beta_trainable: If True, the decoder variance beta is trained
conceptual: the shape of pre-designed prior distribution (circle or star)
```

Step-by-step experiment source code:
```train
MNIST.py: Layers and Models
exon_mnist.py: fitting VAE model for single parameter setting 
exon_mnist_path.py: fitting VAE models for grid parameters setting 
exon_mnist_design.py: fitting VAE models for various pre-designed prior distributions 
```

### 2. CIFAR-10 Dataset

Manual parameter setting:
```
batch_size: mini-batch size of the unlabeled dataset
data_dim: height(or width) of an observation (image)
class_num: the number of labels
latent_dim: dimension size of latent variable
sigma1: diagonal variance element of label-relevant latent dimension part for prior distribution 
sigma2: diagonal variance element of label-irrelevant latent dimension part for prior distribution 
channel: channel size of input image
activation: activation function of output layer of decoder
iterations: the number of total iterations
learning_rate1: learning rate of optimizer for the encoder and the decoder
learning_rate2: learning rate of optimizer for the classifier
hard: If True, Gumbel-Max trick is used at forward pass
lambda1: tuning parameter of classification error
lambda2: tuning parameter of 1/beta regularization if beta is trainable otherwise, it equals to the value of beta
beta_trainable: If True, the observation noise beta is updated by alternating algorithm 
decay_steps: decay step for exponential learning rate decay
decay_rate: decay rate for exponential learning rate decay
labeled: the number of labeled observations
dist: value of diagonal elements which are label-relevant part
ema: If True, exponential moving average is applied to the encoder and decoder weights
slope: slope parameter for leaky ReLU activation in the classifier network
widen_factor: increasing rate for the number of channels for the classifier layers
```

Step-by-step experiemnt source code:
```train
CIFAR10.py: Layers and Models
exon_cifar10.py: fitting VAE model for single parameter setting 
exon_cifar10_path.py: fitting VAE models for grid parameters setting 
exon_cifar10_result.py: get results from fitted models 
```

## Results

### 1. MNIST

#### prior distribution
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/mnist/prior_samples.png?raw=true" width="400"  height="400"></center>

#### posterior plots w.r.t. various tuning parameters
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/mnist/path2.png?raw=true" width="800"  height="600"></center>

### 2. CIFAR-10

#### reconstructed images
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.01/train_recon.png?raw=true" width="400"  height="400"></center>

#### V-nat of EXoN for automobile class when $\beta=0.25$
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.25/vnat.png?raw=true" width="400"  height="180"></center>

#### activated latent subspace determines features of generated image
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/path/lambda2_path.png?raw=true" width="400"  height="200"></center>

<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.25/blur.png?raw=true" width="600"  height="400"></center>

#### interpolation
- $\beta=0.01$
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.01/interpolation.png?raw=true" width="800"  height="150"></center>
- $\beta=0.25$
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.25/interpolation.png?raw=true" width="800"  height="150"></center>


## Citation

```
@misc{an2021exon,
      title={EXoN: EXplainable encoder Network}, 
      author={SeungHwan An and Hosik Choi and Jong-June Jeon},
      year={2021},
      eprint={2105.10867},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```