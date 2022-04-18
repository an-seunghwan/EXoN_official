#%%
import tensorflow as tf
import tensorflow.keras as K

print("TensorFlow version:", tf.__version__)
print("Eager Execution Mode:", tf.executing_eagerly())
print("available GPU:", tf.config.list_physical_devices("GPU"))
from tensorflow.python.client import device_lib

print("==========================================")
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from PIL import Image
import os

os.chdir(r"D:\EXoN")

from modules import MNIST

#%%
PARAMS = {
    "data": "mnist",
    "batch_size": 256,
    "data_dim": 784,
    "class_num": 2,
    "latent_dim": 2,
    "sigma": 4.0,
    "activation": "tanh",
    "iterations": 1000,
    "lambda1": 1000.0,
    "lambda2": 4.0,
    "learning_rate": 0.001,
    "labeled": 1000,
    "hard": True,
    "FashionMNIST": False,
    "beta_trainable": True,
}
#%%
print(
    """
EXoN: EXplainable encoder Network
with MNIST dataset : Latent Space Design
"""
)

pprint(PARAMS)
#%%
# data
if PARAMS["FashionMNIST"]:
    (x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()
else:
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_test = (x_test.astype("float32") - 127.5) / 127.5
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# dataset only from label 0 and 1
label = np.array([0, 1])
train_idx = np.isin(y_train, label)
test_idx = np.isin(y_test, label)

x_train = x_train[train_idx]
x_test = x_test[test_idx]
y_train = y_train[train_idx]
y_test = y_test[test_idx]

from tensorflow.keras.utils import to_categorical

y_train_onehot = to_categorical(y_train, num_classes=PARAMS["class_num"])
y_test_onehot = to_categorical(y_test, num_classes=PARAMS["class_num"])

np.random.seed(520)
# ensure that all classes are balanced
lidx = np.concatenate(
    [
        np.random.choice(
            np.where(y_train == i)[0],
            int(PARAMS["labeled"] / PARAMS["class_num"]),
            replace=False,
        )
        for i in range(PARAMS["class_num"])
    ]
)
x_train_L = x_train[lidx]
y_train_L = y_train_onehot[lidx]

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train))
    .shuffle(len(x_train), reshuffle_each_iteration=True)
    .batch(PARAMS["batch_size"])
)
train_dataset_L = (
    tf.data.Dataset.from_tensor_slices((x_train_L, y_train_L))
    .shuffle(len(x_train_L), reshuffle_each_iteration=True)
    .batch(PARAMS["batch_size"])
)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
    PARAMS["batch_size"]
)
#%%
"""control diversity"""
radius = [4.0, 8.0, 10.0, 12.0]

asset_path = "design"
#%%
for r in radius:
    prior_means = np.array([[-r, 0], [r, 0]])
    prior_means = prior_means[np.newaxis, :, :]
    prior_means = tf.cast(prior_means, tf.float32)
    PARAMS["prior_means"] = prior_means

    model = MNIST.MixtureVAE(PARAMS)
    optimizer = K.optimizers.Adam(PARAMS["learning_rate"])

    @tf.function
    def loss_mixture(prob, xhat, x, mean, logvar, beta, PARAMS):
        # reconstruction error
        error = tf.reduce_mean(
            tf.reduce_sum(tf.math.square(x - xhat), axis=-1) / (2 * beta)
        )

        # KL divergence by closed form
        kl = tf.reduce_mean(
            tf.reduce_sum(prob * tf.math.log(prob * PARAMS["class_num"] + 1e-8), axis=1)
        )
        kl += tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(
                    prob,
                    tf.reduce_sum(
                        0.5
                        * (
                            tf.math.pow(mean - PARAMS["prior_means"], 2)
                            / PARAMS["sigma"]
                            - 1
                            - tf.math.log(1 / PARAMS["sigma"])
                            + tf.math.exp(logvar) / PARAMS["sigma"]
                            - logvar
                        ),
                        axis=-1,
                    ),
                ),
                axis=-1,
            )
        )

        return error, kl

    @tf.function
    def train_step(x_batch_L, y_batch_L, x_batch, beta, lambda1, lambda2):
        with tf.GradientTape() as tape:
            mean, logvar, prob, _, _, _, xhat = model(x_batch)
            error, kl = loss_mixture(prob, xhat, x_batch, mean, logvar, beta, PARAMS)
            loss = error + kl

            prob_L = model.Classifier(x_batch_L)
            cce = -tf.reduce_mean(tf.multiply(y_batch_L, tf.math.log(prob_L + 1e-20)))

            loss += (1.0 + tf.cast(lambda1, tf.float32)) * cce + (
                PARAMS["data_dim"] / 2
            ) * tf.math.log(2 * np.pi * beta)

            if PARAMS["beta_trainable"]:
                loss += (
                    tf.cast(lambda2, tf.float32) * (PARAMS["data_dim"] / 2) * (1 / beta)
                )

        grad = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        return loss, error, kl, cce, xhat

    @tf.function
    def beta_step(x_batch, beta):
        Dz = model(x_batch)[-1]
        beta += tf.reduce_sum(tf.reduce_mean(tf.math.square(x_batch - Dz), axis=-1))
        return beta

    step = 0
    progress_bar = tqdm(range(PARAMS["iterations"]))
    progress_bar.set_description(
        "iteration {}/{} | current loss ?".format(step, PARAMS["iterations"])
    )

    loss_ = []
    error_ = []
    kl_ = []
    cce_ = []

    if PARAMS["beta_trainable"]:
        beta = tf.Variable(
            PARAMS["lambda2"], trainable=True, name="beta", dtype=tf.float32
        )
        betapath = []
    else:
        beta = tf.Variable(
            PARAMS["lambda2"], trainable=False, name="beta", dtype=tf.float32
        )

    for _ in progress_bar:
        x_batch = next(iter(train_dataset))
        x_batch_L, y_batch_L = next(iter(train_dataset_L))

        loss, error, kl, cce, xhat = train_step(
            x_batch_L, y_batch_L, x_batch, beta, PARAMS["lambda1"], PARAMS["lambda2"]
        )

        loss_.append(loss.numpy())
        error_.append(error.numpy())
        kl_.append(kl.numpy())
        cce_.append(cce.numpy())

        if PARAMS["beta_trainable"]:
            # beta training by Alternatning algorithm
            beta = 0
            for x_batch in train_dataset:
                beta = beta_step(x_batch, beta)
            beta = beta / len(x_train) + PARAMS["lambda2"]
            betapath.append(beta.numpy())

        progress_bar.set_description(
            "iteration {}/{} | beta {:.3f}, loss {:.3f}, recon {:.3f}, kl {:.3f}, cce {:.3f}".format(
                step,
                PARAMS["iterations"],
                beta,
                loss_[-1],
                error_[-1],
                kl_[-1],
                cce_[-1],
            )
        )

        step += 1

        if step == PARAMS["iterations"]:
            break

    """reconstruction"""
    grid = np.array([[n, 0] for n in np.linspace(-10, 10, 11)])
    grid_output = model.decoder(tf.cast(np.array(grid), tf.float32))
    grid_output = grid_output.numpy()
    plt.figure(figsize=(8, 4))
    for i in range(len(grid)):
        plt.subplot(1, len(grid), i + 1)
        plt.imshow(grid_output[i].reshape(28, 28), cmap="gray_r")
        plt.axis("off")
        plt.tight_layout()
    plt.savefig(
        "./assets/{}/{}/reconstruction_radius_{}.png".format(
            PARAMS["data"], asset_path, r
        ),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()

    """posterior"""
    zmat = []
    mean, logvar, logits, y, z, z_tilde, xhat = model(x_test)
    zmat.extend(z_tilde.numpy().reshape(-1, PARAMS["latent_dim"]))
    zmat = np.array(zmat)
    plt.figure(figsize=(7, 7))
    plt.tick_params(labelsize=10)
    plt.scatter(zmat[:, 0], zmat[:, 1], c=y_test, s=10, alpha=1)
    plt.savefig(
        "./assets/{}/{}/latentspace_{}.png".format(PARAMS["data"], asset_path, r),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # plt.show()
    plt.close()

    del model
    tf.keras.backend.clear_session()
#%%
"""Appendix: Figure 2"""
reconstruction = []
for r in radius:
    reconstruction.append(
        Image.open(
            "./assets/{}/{}/reconstruction_radius_{}.png".format(
                PARAMS["data"], asset_path, r
            )
        )
    )
plt.figure(figsize=(10, 5))
for i in range(len(reconstruction)):
    plt.subplot(4, 1, i + 1)
    plt.imshow(reconstruction[i], cmap="gray_r")
    plt.axis("off")
    plt.tight_layout()
plt.savefig(
    "./assets/{}/{}/reconstruction.png".format(PARAMS["data"], asset_path),
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.1,
)
# plt.show()
plt.close()

reconstruction = Image.open(
    "./assets/{}/{}/reconstruction.png".format(PARAMS["data"], asset_path)
)

plt.figure(figsize=(10, 10))
plt.xticks(np.linspace(-10, 10, 11))
plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=0)
plt.imshow(reconstruction, extent=[-11.35, 11.2, -5, 5])
plt.tight_layout()
plt.savefig(
    "./assets/{}/{}/reconstruction.png".format(PARAMS["data"], asset_path),
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.1,
)
# plt.show()
plt.close()
#%%
