#%%
import argparse
import os

os.chdir(r'D:\EXoN_official') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/EXoN_official') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.utils import to_categorical
import tqdm
import yaml
import io
import matplotlib.pyplot as plt

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import MixtureVAE
from criterion import ELBO_criterion
from mixup import augment, non_smooth_mixup, weight_decay_decoupled
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

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset used for training (only mnist)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--labeled-batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=100, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=100, 
                        help='number labeled examples (default: 100), all labels are balanced')
    # parser.add_argument('--validation_examples', type=int, default=5000, 
    #                     help='number validation examples (default: 5000')
    parser.add_argument('--augment', default=True, type=bool,
                        help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    parser.add_argument('--bce_reconstruction', default=False, type=bool,
                        help="Do BCE Reconstruction")
    # parser.add_argument('--beta_trainable', default=False, type=bool,
    #                     help="trainable beta")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=2, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma', default=4, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--radius', default=4, type=float,  
                        help='center coordinate of pre-designed components: (-r, 0), (r, 0)')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--lambda1', default=6000, type=int, # labeled dataset ratio?
                        help='the weight of classification loss term')
    parser.add_argument('--beta', default=1, type=int, 
                        help='value of beta (observation noise)')
    parser.add_argument('--rampup_epoch',default=10, type=int, 
                        help='the max epoch to adjust learning rate and unsupervised weight')
    parser.add_argument('--rampdown_epoch',default=10, type=int, 
                        help='the last epoch to adjust learning rate')
    
    '''Optimizer Parameters'''
    parser.add_argument('--learning_rate', default=3e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    '''Interpolation Parameters'''
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="beta distribution parameter")

    '''Configuration'''
    parser.add_argument('--config_path', type=str, default=None, 
                        help='path to yaml config file, overwrites args')

    return parser.parse_args()
#%%
def load_config(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args['config_path'])    
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args
#%%
def generate_and_save_images1(model, image, num_classes):
    _, _, _, _, z, _ = model.encode(image, training=False)
    
    buf = io.BytesIO()
    figure = plt.figure(figsize=(10, 2))
    plt.subplot(1, num_classes+1, 1)
    # plt.imshow(image[0])
    plt.imshow((image[0] + 1) / 2)
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        xhat = model.decode(z.numpy()[0][[i]], training=False)
        plt.subplot(1, num_classes+1, i+2)
        # plt.imshow(xhat[0])
        plt.imshow((xhat[0] + 1) / 2)
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=1)
    image = tf.expand_dims(image, 0)
    return image

def generate_and_save_images2(model, image, num_classes, step, save_dir):
    _, _, _, _, z, _ = model.encode(image, training=False)
    
    plt.figure(figsize=(10, 2))
    plt.subplot(1, num_classes+1, 1)
    # plt.imshow(image[0])
    plt.imshow((image[0] + 1) / 2)
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        xhat = model.decode(z.numpy()[0][[i]], training=False)
        plt.subplot(1, num_classes+1, i+2)
        # plt.imshow(xhat[0])
        plt.imshow((xhat[0] + 1) / 2)
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig('{}/image_at_epoch_{}.png'.format(save_dir, step))
    # plt.show()
    plt.close()
#%%
def main():
    '''argparse to dictionary'''
    args = vars(get_args())
    # '''argparse debugging'''
    # args = vars(parser.parse_args(args=[]))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_pre_design'

    '''dataset'''
    (x_train, y_train), (_, _) = K.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") - 127.5) / 127.5
    num_classes = 10

    # dataset only from label 0 and 1
    label = np.array([0, 1])
    train_idx = np.isin(y_train, label)

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    np.random.seed(1)
    # ensure that all classes are balanced
    lidx = np.concatenate(
        [
            np.random.choice(
                np.where(y_train == i)[0],
                int(args["labeled_examples"] / num_classes),
                replace=False,
            )
            for i in [0, 1]
        ]
    )
    x_train_L = x_train[lidx]
    y_train_L = to_categorical(y_train[lidx], num_classes=num_classes)
    
    datasetU = (
        tf.data.Dataset.from_tensor_slices((x_train))
        .shuffle(len(x_train), reshuffle_each_iteration=True)
        .batch(args["batch_size"])
    )
    datasetL = (
        tf.data.Dataset.from_tensor_slices((x_train_L, y_train_L))
        .shuffle(len(x_train_L), reshuffle_each_iteration=True)
        .batch(args["labeled_batch_size"])
    )
    total_length = sum(1 for _ in datasetU)
    
    model = MixtureVAE(args,
                    num_classes,
                    latent_dim=args['latent_dim'])
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()
    
    buffer_model = MixtureVAE(args,
                            num_classes,
                            latent_dim=args['latent_dim'])
    buffer_model.build(input_shape=(None, 28, 28, 1))
    buffer_model.set_weights(model.get_weights()) # weight initialization
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['learning_rate'])
    optimizer_classifier = K.optimizers.Adam(learning_rate=args['learning_rate'])

    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    '''prior design'''
    prior_means = np.array([[-args['radius'], 0], [args['radius'], 0]])
    prior_means = prior_means[np.newaxis, :, :]
    prior_means = tf.cast(prior_means, tf.float32)
    sigma = tf.cast(args['sigma'], tf.float32)
    
    '''initialize beta'''
    beta = tf.cast(args['beta'], tf.float32) 
    
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''classifier: learning rate schedule'''
        if epoch >= args['rampdown_epoch']:
            optimizer_classifier.lr = args['lr'] * tf.math.exp(-5 * (1. - (args['epochs'] - epoch) / args['epochs']) ** 2)
            optimizer_classifier.beta_1 = 0.5
            
        if epoch % args['reconstruct_freq'] == 0:
            loss, recon_loss, kl1_loss, kl2_loss, label_mixup_loss, unlabel_mixup_loss, accuracy, sample_recon = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, prior_means, sigma, num_classes, total_length, test_accuracy_print)
        else:
            loss, recon_loss, kl1_loss, kl2_loss, label_mixup_loss, unlabel_mixup_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, prior_means, sigma, num_classes, total_length, test_accuracy_print)
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('recon_loss', recon_loss.result(), step=epoch)
            tf.summary.scalar('kl1', kl1_loss.result(), step=epoch)
            tf.summary.scalar('kl2', kl2_loss.result(), step=epoch)
            tf.summary.scalar('label_mixup_loss', label_mixup_loss.result(), step=epoch)
            tf.summary.scalar('unlabel_mixup_loss', unlabel_mixup_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
            if epoch % args['reconstruct_freq'] == 0:
                tf.summary.image("train recon image", sample_recon, step=epoch)

        # Reset metrics every epoch
        loss.reset_states()
        recon_loss.reset_states()
        kl1_loss.reset_states()
        kl2_loss.reset_states()
        label_mixup_loss.reset_states()
        unlabel_mixup_loss.reset_states()
        accuracy.reset_states()
        
        # if args['beta_trainable']:
        #     '''beta update'''
        #     beta = update_beta(model, datasetU, args, total_length)
            
    '''model & configurations save'''        
    # weight name for saving
    for i, w in enumerate(model.variables):
        split_name = w.name.split('/')
        if len(split_name) == 1:
            new_name = split_name[0] + '_' + str(i)    
        else:
            new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i)
        model.variables[i]._handle_name = new_name
    
    model_path = f'{log_path}/{current_time}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model.save_weights(model_path + '/model_{}.h5'.format(current_time), save_format="h5")

    with open(model_path + '/args_{}.txt'.format(current_time), "w") as f:
        for key, value, in args.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
#%%
def train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, prior_means, sigma, num_classes, total_length, test_accuracy_print):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    kl1_loss_avg = tf.keras.metrics.Mean()
    kl2_loss_avg = tf.keras.metrics.Mean()
    label_mixup_loss_avg = tf.keras.metrics.Mean()
    unlabel_mixup_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    '''supervised classification weight'''
    lambda1 = tf.cast(args['lambda1'], tf.float32)
    '''un-supervised reconstruction weight'''
    lambda2 = weight_schedule(epoch, args['rampup_epoch'], lambda1)
    
    autotune = tf.data.AUTOTUNE
    shuffle_and_batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e3)).batch(batch_size=32, 
                                                                                    drop_remainder=False).prefetch(autotune)
    shuffle_and_batchU = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'] - 32, 
                                                                                    drop_remainder=True).prefetch(autotune)

    iteratorL = iter(shuffle_and_batchL(datasetL))
    iteratorU = iter(shuffle_and_batchU(datasetU))
        
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batchL(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batchU(datasetU))
            imageU, _ = next(iteratorU)
        
        if args['augment']:
            imageL_aug = augment(imageL)
            imageU_aug = augment(imageU)
            
        # non-augmented image
        image = tf.concat([imageL, imageU], axis=0) 
            
        '''mix-up weight'''
        mix_weight = [tf.constant(np.random.beta(args['epsilon'], args['epsilon'])), # labeled
                      tf.constant(np.random.beta(2.0, 2.0))] # unlabeled
            
        with tf.GradientTape(persistent=True) as tape:    
            '''ELBO'''
            mean, logvar, prob, y, z, z_tilde, xhat = model(image)
            recon_loss, kl1, kl2 = ELBO_criterion(prob, xhat, image, mean, logvar, 
                                                prior_means, sigma, num_classes, args)
            probL_aug = model.classify(imageL_aug)
            cce = - tf.reduce_sum(tf.reduce_sum(tf.multiply(labelL, tf.math.log(tf.clip_by_value(probL_aug, 1e-10, 1.))), axis=-1))
            
            '''consistency interpolation'''
            # mix-up
            with tape.stop_recording():
                image_mixL, label_shuffleL = non_smooth_mixup(imageL_aug, labelL, mix_weight[0])
                # classifier output of right before epoch
                pseudo_labelU = buffer_model.classify(imageU_aug)    
                image_mixU, pseudo_label_shuffleU = non_smooth_mixup(imageU_aug, pseudo_labelU, mix_weight[1])
            # labeled
            prob_mixL = model.classify(image_mixL)
            mixup_yL = - tf.reduce_sum(mix_weight[0] * tf.reduce_sum(label_shuffleL * tf.math.log(tf.clip_by_value(prob_mixL, 1e-10, 1.0)), axis=-1))
            mixup_yL += - tf.reduce_sum((1. - mix_weight[0]) * tf.reduce_sum(labelL * tf.math.log(tf.clip_by_value(prob_mixL, 1e-10, 1.0)), axis=-1))
            # unlabeled
            prob_mixU = model.classify(image_mixU)
            mixup_yU = - tf.reduce_sum(mix_weight[1] * tf.reduce_sum(pseudo_label_shuffleU * tf.math.log(tf.clip_by_value(prob_mixU, 1e-10, 1.0)), axis=-1))
            mixup_yU += - tf.reduce_sum((1. - mix_weight[1]) * tf.reduce_sum(pseudo_labelU * tf.math.log(tf.clip_by_value(prob_mixU, 1e-10, 1.0)), axis=-1))
            
            elbo = recon_loss + beta * (kl1 + kl2 + cce)

            loss = elbo + lambda1 * (cce + mixup_yL) + lambda2 * mixup_yU
        
        # encoder and decoder
        grads = tape.gradient(loss, model.decoder.trainable_variables + model.encoder.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.decoder.trainable_variables + model.encoder.trainable_variables)) 
        # classifier
        grads = tape.gradient(loss, model.classifier.trainable_variables) 
        optimizer_classifier.apply_gradients(zip(grads, model.classifier.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model.classifier, buffer_model.classifier, decay_rate=args['weight_decay'] * optimizer_classifier.lr)
        
        loss_avg(loss)
        recon_loss_avg(recon_loss / args['batch_size'])
        kl1_loss_avg(kl1 / args['batch_size'])
        kl2_loss_avg(kl2 / args['batch_size'])
        label_mixup_loss_avg(mixup_yL / args['batch_size'])
        unlabel_mixup_loss_avg(mixup_yU / args['batch_size'])
        probL = model.classify(imageL, training=False)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'Recon': f'{recon_loss_avg.result():.4f}',
            'KL1': f'{kl1_loss_avg.result():.4f}',
            'KL2': f'{kl2_loss_avg.result():.4f}',
            'MixUp(L)': f'{label_mixup_loss_avg.result():.4f}',
            'MixUp(U)': f'{unlabel_mixup_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
            'beta': f'{beta:.4f}'
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images1(model, imageU[0][tf.newaxis, ...], num_classes)
        generate_and_save_images2(model, imageU[0][tf.newaxis, ...], num_classes, epoch, f'logs/{args["dataset"]}_{args["labeled_examples"]}/{current_time}')
        return loss_avg, recon_loss_avg, kl1_loss_avg, kl2_loss_avg, label_mixup_loss_avg, unlabel_mixup_loss_avg, accuracy, sample_recon
    else:
        return loss_avg, recon_loss_avg, kl1_loss_avg, kl2_loss_avg, label_mixup_loss_avg, unlabel_mixup_loss_avg, accuracy
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
# def update_beta(model, datasetU, args, total_length):
#     autotune = tf.data.AUTOTUNE
#     batch = lambda dataset: dataset.batch(batch_size=args['batch_size'], drop_remainder=False).prefetch(autotune)
#     iteratorU = iter(batch(datasetU))
    
#     iteration = total_length // args['batch_size'] 
#     beta = 0.
#     for _ in range(iteration + 1):
#         imageU, _ = next(iteratorU)
#         xhatU = model(imageU)[-1]
#         beta += tf.reduce_sum(tf.reduce_mean(tf.math.square(xhatU - imageU), axis=-1))
#     beta = beta / 60000. + tf.cast(args['beta'], tf.float32)
#     return beta
#%%
if __name__ == '__main__':
    main()
#%%