# EXoN: EXplainable encoder Network

This repository is the official implementation of [EXoN: EXplainable encoder Network](XXX) with Tensorflow 2.0. 

## Package Dependencies

```setup
python==3.7
numpy==1.19.5
tensorflow==2.4.0
```
Additional package requirements for this repository are described in `requirements.txt`.

## Training & Evaluation 

### 0. directory and codes

```
.
+-- logs (folder which contains results of all experiments)
|       +-- mnist_100 (with 100 labeled datasets)
|       +-- mnist_pre_design (Appendix A.2.2)
|       +-- cifar10_4000 (with 4000 labeled datasets)

+-- mnist (folder which contains source codes for MNIST dataset experiments)
|   +-- configs (folder which contains configuration informations of MNIST dataset experiments)
|       +-- mnist_100.yaml
|   +-- preprocess.py
|   +-- model.py
|   +-- criterion.py
|   +-- mixup.py
|   +-- main.py
|   +-- result.py
|   +-- pre_design.py (Appendix A.2.2)
|   +-- pre_design_result.py (Appendix A.2.2)

(source codes for CIFAR-10 dataset experiments)
+-- configs (folder which contains configuration informations of CIFAR-10 dataset experiments)
|       +-- cifar10_4000.yaml
+-- preprocess.py
+-- model.py
+-- criterion.py
+-- mixup.py
+-- main.py
+-- result.py
+-- README.md
+-- requirements.txt
+-- LICENSE
```

### 1. How to Training & Evaluation  

`labeled_examples` is the number of labeled datsets for running and we provide configuration `.yaml` files for 100 labeled datsets of MNIST and 4000 labeled datasets of CIFAR-10. And we add required tests and evaluations at the end of code.

1. MNIST datset experiment

```
python main.py --config_path "configs/mnist_{labeled_examples}.yaml"
```   
- for Appedix A.2.2,
```
python pre_design.py --config_path "configs/mnist_{labeled_examples}.yaml"
```

To get visualization results of MNIST dataset experiment and Appendix A.2.2, use `mnist/result.py` and `mnist/pre_design_result.py` (following codes line by line).

2. CIFAR-10 dataset experiment

```
python main.py --config_path "configs/cifar10_{labeled_examples}.yaml"
```

To get visualization results of CIFAR-10 dataset experiment, use `result.py` (following the code line by line).

## Results

### 1. MNIST

<!-- #### prior distribution
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/mnist/prior_samples.png?raw=true" width="400"  height="400"></center> -->

#### Effect of Regularizations
<center><img  src="https://github.com/an-seunghwan/EXoN_official/blob/main/logs/mnist_100/path_latent_recon.png?raw=true" width="800"  height="400"></center>

### 2. CIFAR-10

<!-- #### reconstructed images
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.01/train_recon.png?raw=true" width="400"  height="400"></center>

#### V-nat of EXoN for automobile class when $\beta=0.25$
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.25/vnat.png?raw=true" width="400"  height="180"></center> -->

#### The activated latent subspace determines features of generated image
<center><img  src="https://github.com/an-seunghwan/EXoN_official/blob/main/logs/cifar10_4000/beta_0.05/blur.png?raw=true" width="600"  height="400"></center>

<!-- <center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.25/blur.png?raw=true" width="600"  height="400"></center> -->

#### interpolation
- between same classes
<center><img  src="https://github.com/an-seunghwan/EXoN_official/blob/main/logs/cifar10_4000/beta_0.05/interpolation1.png?raw=true" width="800"  height="150"></center>

- between different classes
<center><img  src="https://github.com/an-seunghwan/EXoN_official/blob/main/logs/cifar10_4000/beta_0.05/interpolation2.png?raw=true" width="800"  height="150"></center>

<!-- - $\beta=0.25$
<center><img  src="https://github.com/an-seunghwan/EXoN/blob/main/assets/cifar10/weights_10000.0_0.25/interpolation.png?raw=true" width="800"  height="150"></center> -->


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