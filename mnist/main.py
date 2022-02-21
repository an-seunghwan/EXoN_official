#%%
'''
***check list!***
- perfectly white background

- the number of epochs
- decoupled encoder
    -> DropOut classifier
- mutual information bound on discrete kl-divergence
- without augmentation
- consistency interpolation is added
- loss weight on consistency interpolation (unlabeled)
'''
#%%
import argparse
import os

os.chdir(r'D:\EXoN_official') # main directory (repository)
# os.chdir('/home1/prof/jeon/an/semi/semi/proposal') # main directory (repository)

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm
import yaml
import io
import matplotlib.pyplot as plt

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from preprocess import fetch_dataset
from model import MixtureVAE
from criterion import ELBO_criterion
from mixup import augment, label_smoothing, non_smooth_mixup, weight_decay_decoupled
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
                        help='dataset used for training')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=100, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=100, 
                        help='number labeled examples (default: 100')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    parser.add_argument('--augment', default=False, type=bool,
                        help="apply augmentation to image")

    # '''Deep VAE Model Parameters'''
    # parser.add_argument('--depth', type=int, default=28, 
    #                     help='depth for WideResnet (default: 28)')
    # parser.add_argument('--width', type=int, default=2, 
    #                     help='widen factor for WideResnet (default: 2)')
    # parser.add_argument('--slope', type=float, default=0.1, 
    #                     help='slope parameter for LeakyReLU (default: 0.1)')
    # parser.add_argument('-dr', '--drop_rate', default=0, type=float, 
    #                     help='drop rate for the network')
    # parser.add_argument("--br", "--bce_reconstruction", action='store_true', 
    #                     help='Do BCE Reconstruction')
    # parser.add_argument("-s", "--x_sigma", default=0.5, type=float,
    #                     help="The standard variance for reconstructed images, work as regularization")

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=2, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma', default=4, type=float,  
                        help='variance of prior mixture component')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--kl_y_threshold', default=2.3, type=float,  
                        help='mutual information bound of discrete kl-divergence')
    parser.add_argument('--lambda1',default=1000, type=int, 
                        help='the weight of classification loss term')
    parser.add_argument('--lambda2',default=4, type=int, 
                        help='the weight of beta penalty term')
    parser.add_argument('--mixup_max_y', default=10, type=float, 
                        help='the max value for mixup(y) weight')
    parser.add_argument('--mixup_epoch_y',default=50, type=int, 
                        help='the max epoch to adjust mixup')
    # parser.add_argument('--recon_max', default=1, type=float, 
    #                     help='the max value for reconstruction error')
    # parser.add_argument('--recon_max_epoch',default=1, type=int, 
    #                     help='the max epoch to adjust reconstruction error')
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-ad', "--adjust_lr", default=[75, 90], type=arg_as_list,
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--wd', '--weight_decay', default=5e-4, type=float)

    '''Optimizer Transport Estimation Parameters'''
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help="the label smoothing epsilon for labeled data")

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
    plt.imshow(image[0])
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        xhat = model.decode(z.numpy()[0][[i]], training=False)
        plt.subplot(1, num_classes+1, i+2)
        plt.imshow(xhat[0])
        plt.title('{}'.format(i))
        plt.axis('off')
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=2)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def generate_and_save_images2(model, image, num_classes, step, save_dir):
    _, _, _, _, z, _ = model.encode(image, training=False)
    
    plt.figure(figsize=(10, 2))
    plt.subplot(1, num_classes+1, 1)
    plt.imshow(image[0])
    plt.title('original')
    plt.axis('off')
    for i in range(num_classes):
        xhat = model.decode(z.numpy()[0][[i]], training=False)
        plt.subplot(1, num_classes+1, i+2)
        plt.imshow(xhat[0])
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
    # args = vars(parser.parse_args(args=['--config_path', 'configs/mnist_100.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    total_length = sum(1 for _ in datasetU)
    
    model = MixtureVAE(args,
                    num_classes,
                    latent_dim=args['latent_dim'])
    model.build(input_shape=(None, 32, 32))
    model.summary()
    
    buffer_model = MixtureVAE(args,
                    num_classes,
                    latent_dim=args['latent_dim'])
    buffer_model.build(input_shape=(None, 32, 32))
    buffer_model.set_weights(model.get_weights()) # weight initialization
    
    '''optimizer'''
    optimizer = K.optimizers.Adam(learning_rate=args['lr'])
    
    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    test_accuracy_print = 0.
    
    '''prior design'''
    r = 2. * np.sqrt(args['sigma']) / np.sin(np.pi / 10.)
    prior_means = np.array([[r*np.cos(np.pi/10), r*np.sin(np.pi/10)],
                            [r*np.cos(3*np.pi/10), r*np.sin(3*np.pi/10)],
                            [r*np.cos(5*np.pi/10), r*np.sin(5*np.pi/10)],
                            [r*np.cos(7*np.pi/10), r*np.sin(7*np.pi/10)],
                            [r*np.cos(9*np.pi/10), r*np.sin(9*np.pi/10)],
                            [r*np.cos(11*np.pi/10), r*np.sin(11*np.pi/10)],
                            [r*np.cos(13*np.pi/10), r*np.sin(13*np.pi/10)],
                            [r*np.cos(15*np.pi/10), r*np.sin(15*np.pi/10)],
                            [r*np.cos(17*np.pi/10), r*np.sin(17*np.pi/10)],
                            [r*np.cos(19*np.pi/10), r*np.sin(19*np.pi/10)]])
    prior_means = tf.cast(prior_means[np.newaxis, :, :], tf.float32)
    
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        if epoch == 0:
            '''warm-up'''
            optimizer.lr = args['lr'] * 0.2
        elif epoch < args['adjust_lr'][0]:
            optimizer.lr = args['lr']
        elif epoch < args['adjust_lr'][1]:
            optimizer.lr = args['lr'] * args['lr_gamma']
        else:
            optimizer.lr = args['lr'] * (args['lr_gamma'] ** 2)
            
        if epoch % args['reconstruct_freq'] == 0:
            loss, recon_loss, kl1_loss, kl2_loss, mixup_yU_loss, mixup_yL_loss, accuracy, sample_recon = train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, prior_means, sigma_vector, num_classes, total_length, test_accuracy_print)
        else:
            loss, recon_loss, kl1_loss, kl2_loss, mixup_yU_loss, mixup_yL_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, prior_means, sigma_vector, num_classes, total_length, test_accuracy_print)
        # loss, recon_loss, info_loss, nf_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_nf, epoch, args, num_classes, total_length)
        val_recon_loss, val_kl1_loss, val_kl2_loss, val_elbo_loss, val_accuracy = validate(val_dataset, model, epoch, args, prior_means, sigma_vector, num_classes, split='Validation')
        test_recon_loss, test_kl1_loss, test_kl2_loss, test_elbo_loss, test_accuracy = validate(test_dataset, model, epoch, args, prior_means, sigma_vector, num_classes, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('recon_loss', recon_loss.result(), step=epoch)
            tf.summary.scalar('kl1', kl1_loss.result(), step=epoch)
            tf.summary.scalar('kl2', kl2_loss.result(), step=epoch)
            tf.summary.scalar('mixup_yU_loss', mixup_yU_loss.result(), step=epoch)
            tf.summary.scalar('mixup_yL_loss', mixup_yL_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy.result(), step=epoch)
            if epoch % args['reconstruct_freq'] == 0:
                tf.summary.image("train recon image", sample_recon, step=epoch)
        with val_writer.as_default():
            tf.summary.scalar('recon_loss', val_recon_loss.result(), step=epoch)
            tf.summary.scalar('val_kl1_loss', val_kl1_loss.result(), step=epoch)
            tf.summary.scalar('val_kl2_loss', val_kl2_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', val_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar('recon_loss', test_recon_loss.result(), step=epoch)
            tf.summary.scalar('test_kl1_loss', test_kl1_loss.result(), step=epoch)
            tf.summary.scalar('test_kl2_loss', test_kl2_loss.result(), step=epoch)
            tf.summary.scalar('elbo_loss', test_elbo_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            
        test_accuracy_print = test_accuracy.result()

        # Reset metrics every epoch
        loss.reset_states()
        recon_loss.reset_states()
        kl1_loss.reset_states()
        kl2_loss.reset_states()
        mixup_yU_loss.reset_states()
        mixup_yL_loss.reset_states()
        accuracy.reset_states()
        val_recon_loss.reset_states()
        val_kl1_loss.reset_states()
        val_kl2_loss.reset_states()
        val_elbo_loss.reset_states()
        val_accuracy.reset_states()
        test_recon_loss.reset_states()
        test_kl1_loss.reset_states()
        test_kl2_loss.reset_states()
        test_elbo_loss.reset_states()
        test_accuracy.reset_states()
        
        if epoch == 0:
            optimizer.lr = args['lr']
            
        # if args['dataset'] == 'cifar10':
        #     if args['labeled_examples'] >= 2500:
        #         if epoch == args['adjust_lr'][0]:
        #             args['recon_max'] = args['recon_max'] * 5

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
def train(datasetL, datasetU, model, buffer_model, optimizer, epoch, args, prior_means, sigma_vector, num_classes, total_length, test_accuracy_print):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    kl1_loss_avg = tf.keras.metrics.Mean()
    kl2_loss_avg = tf.keras.metrics.Mean()
    mixup_yU_loss_avg = tf.keras.metrics.Mean()
    mixup_yL_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    '''un-supervised classification weight'''
    mixup_lambda_y = weight_schedule(epoch, args['mixup_epoch_y'], args['mixup_max_y'])
    '''supervised classification weight'''
    lambda1 = tf.cast(args['lambda1'], tf.float32)
    '''trainable beta'''
    beta = tf.cast(args['lambda2'], tf.float32) # initial value

    autotune = tf.data.AUTOTUNE
    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)

    iteratorL = iter(shuffle_and_batch(datasetL))
    iteratorU = iter(shuffle_and_batch(datasetU))
        
    # iteration = (50000 - args['validation_examples']) // args['batch_size'] 
    iteration = total_length // args['batch_size'] 
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        try:
            imageL, labelL = next(iteratorL)
        except:
            iteratorL = iter(shuffle_and_batch(datasetL))
            imageL, labelL = next(iteratorL)
        try:
            imageU, _ = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            imageU, _ = next(iteratorU)
        
        if args['augment']:
            imageL = augment(imageL)
            imageU = augment(imageU)
        
        '''mix-up weight'''
        mix_weight = [tf.constant(np.random.beta(args['epsilon'], args['epsilon'])), # labeled
                      tf.constant(np.random.beta(2.0, 2.0))] # unlabeled
        
        with tf.GradientTape(persistent=True) as tape:
            '''ELBO'''
            meanU, logvarU, probU, yU, zU, z_tildeU, xhatU = model(imageU)
            recon_lossU, kl1U, kl2U = ELBO_criterion(probU, xhatU, imageU, meanU, logvarU, 
                                                    beta, prior_means, sigma_vector, num_classes, args)
            probL = model.classify(imageL)
            cce = - tf.reduce_mean(tf.reduce_sum(tf.multiply(labelL, tf.math.log(tf.clip_by_value(probL, 1e-10, 1.))), axis=-1))
            
            '''mix-up: consistency interpolation'''
            with tape.stop_recording():
                image_mixU, prob_mixU = non_smooth_mixup(imageU, probU, mix_weight[1])
                image_mixL, label_mixL, label_shuffleL = label_smoothing(imageL, labelL, mix_weight[0])
            
            smoothed_probU = model.classify(image_mixU)
            # '''CE'''
            # mixup_yU = - tf.reduce_mean(tf.reduce_sum(prob_mixU * tf.math.log(tf.clip_by_value(smoothed_probU, 1e-10, 1.0)), axis=-1))
            '''JSD'''
            mixup_yU = 0.5 * tf.reduce_mean(tf.reduce_sum(prob_mixU * (tf.math.log(tf.clip_by_value(prob_mixU, 1e-10, 1.0)) - 
                                                                       tf.math.log(tf.clip_by_value(smoothed_probU, 1e-10, 1.0))), axis=1))
            mixup_yU += 0.5 * tf.reduce_mean(tf.reduce_sum(smoothed_probU * (tf.math.log(tf.clip_by_value(smoothed_probU, 1e-10, 1.0)) - 
                                                                             tf.math.log(tf.clip_by_value(prob_mixU, 1e-10, 1.0))), axis=1))
            
            smoothed_probL = model.classify(image_mixL)
            mixup_yL = - tf.reduce_mean(mix_weight[0] * tf.reduce_sum(label_shuffleL * tf.math.log(tf.clip_by_value(smoothed_probL, 1e-10, 1.0)), axis=-1))
            mixup_yL += - tf.reduce_mean((1. - mix_weight[0]) * tf.reduce_sum(labelL * tf.math.log(tf.clip_by_value(smoothed_probL, 1e-10, 1.0)), axis=-1))
            
            ELBO = recon_lossU + kl1U + kl2U
            lossU = ELBO + (mixup_lambda_y * mixup_yU)
            lossL = (1. + lambda1) * (cce + mixup_yL)
            loss = lossU + lossL + (32 * 32 / 2) * tf.math.log(2. * np.pi * beta)

        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model, buffer_model, decay_rate=args['wd'] * optimizer.lr)
        
        '''beta update'''
        beta = update_beta(model, datasetU, args, total_length)
        
        loss_avg(loss)
        recon_loss_avg(recon_lossU)
        kl1_loss_avg(kl1U)
        kl2_loss_avg(kl2U)
        mixup_yU_loss_avg(mixup_yU)
        mixup_yL_loss_avg(mixup_yL)
        probL = model.classify(imageL, training=False)
        accuracy(tf.argmax(labelL, axis=1, output_type=tf.int32), probL)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'Recon': f'{recon_loss_avg.result():.4f}',
            'KL1': f'{kl1_loss_avg.result():.4f}',
            'KL2': f'{kl2_loss_avg.result():.4f}',
            'MIXUP(U)': f'{mixup_yU_loss_avg.result():.4f}',
            'MIXUP(L)': f'{mixup_yL_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}'
        })
        
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images1(model, imageU[0][tf.newaxis, ...], num_classes)
        generate_and_save_images2(model, imageU[0][tf.newaxis, ...], num_classes, epoch, f'logs/{args["dataset"]}_{args["labeled_examples"]}/{current_time}')
        return loss_avg, recon_loss_avg, kl1_loss_avg, kl2_loss_avg, mixup_yU_loss_avg, mixup_yL_loss_avg, accuracy, sample_recon
    else:
        return loss_avg, recon_loss_avg, kl1_loss_avg, kl2_loss_avg, mixup_yU_loss_avg, mixup_yL_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, prior_means, sigma_vector, num_classes, split):
    nf_loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()   
    kl1_loss_avg = tf.keras.metrics.Mean()   
    kl2_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()
    
    beta = 1.

    dataset = dataset.batch(args['batch_size'])
    for image, label in dataset:
        mean, logvar, prob, y, z, z_tilde, xhat = model(image, training=False)
        recon_loss, kl1, kl2 = ELBO_criterion(prob, xhat, image, mean, logvar, 
                                                beta, prior_means, sigma_vector, num_classes, args)
        cce = - tf.reduce_mean(tf.reduce_sum(tf.multiply(label, tf.math.log(tf.clip_by_value(prob, 1e-10, 1.))), axis=-1))
        
        recon_loss_avg(recon_loss)
        kl1_loss_avg(kl1)
        kl2_loss_avg(kl2)
        elbo_loss_avg(recon_loss + kl1 + kl2 + cce)
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, RECON: {recon_loss_avg.result():.4f}, KL1: {kl1_loss_avg.result():.4f}, KL2: {kl2_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return recon_loss_avg, kl1_loss_avg, kl2_loss_avg, elbo_loss_avg, accuracy
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
def update_beta(model, datasetU, args, total_length):
    autotune = tf.data.AUTOTUNE
    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], 
                                                                                    drop_remainder=True).prefetch(autotune)
    iteratorU = iter(shuffle_and_batch(datasetU))
    
    beta = 0.
    for _ in range(total_length):
        imageU, _ = next(iteratorU)
        xhatU = model(imageU)[-1]
        beta += tf.reduce_sum(tf.reduce_mean(tf.math.square(xhatU - imageU), axis=-1))
        return beta
#%%
if __name__ == '__main__':
    main()
#%%