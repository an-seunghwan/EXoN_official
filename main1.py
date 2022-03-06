#%%
'''
***check list!***
- Stochastic Weight Averaging
- first 10-dimensions of latent mean vector = sigmoid activation
- only weight decay on classifier
- pseudo label = training False
- without jitter -> normalization necessary
- additional classification regularization term weight = (lambda / beta) * beta
'''
#%%
import argparse
import os

# os.chdir(r'D:\EXoN_official') # main directory (repository)
os.chdir('/home1/prof/jeon/an/EXoN_official') # main directory (repository)

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

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used for training (e.g. cmnist, cifar10, svhn, svhn+extra)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    '''SSL VAE Train PreProcess Parameter'''
    parser.add_argument('--epochs', default=400, type=int, 
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, 
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--reconstruct_freq', '-rf', default=10, type=int,
                        metavar='N', help='reconstruct frequency (default: 10)')
    parser.add_argument('--labeled_examples', type=int, default=4000, 
                        help='number labeled examples (default: 4000')
    parser.add_argument('--validation_examples', type=int, default=5000, 
                        help='number validation examples (default: 5000')
    parser.add_argument('--augment', default=True, type=bool,
                        help="apply augmentation to image")

    '''Deep VAE Model Parameters'''
    parser.add_argument('-dr', '--drop_rate', default=0, type=float, 
                        help='drop rate for the network')
    parser.add_argument('--bce', "--bce_reconstruction", default=False, type=bool,
                        help="Do BCE Reconstruction")
    parser.add_argument('--beta_trainable', default=False, type=bool,
                        help="trainable beta")
    # parser.add_argument('--depth', type=int, default=28, 
    #                     help='depth for WideResnet (default: 28)')
    # parser.add_argument('--width', type=int, default=2, 
    #                     help='widen factor for WideResnet (default: 2)')
    # parser.add_argument('--slope', type=float, default=0.1, 
    #                     help='slope parameter for LeakyReLU (default: 0.1)')

    '''VAE parameters'''
    parser.add_argument('--latent_dim', "--latent_dim_continuous", default=256, type=int,
                        metavar='Latent Dim For Continuous Variable',
                        help='feature dimension in latent space for continuous variable')
    
    '''Prior design'''
    parser.add_argument('--sigma1', default=0.1, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--sigma2', default=1, type=float,  
                        help='variance of prior mixture component')
    parser.add_argument('--dist', default=1, type=float,  
                        help='first 10-dimension latent mean vector value')

    '''VAE Loss Function Parameters'''
    parser.add_argument('--kl_y_threshold', default=0, type=float,  
                        help='mutual information bound of discrete kl-divergence')
    parser.add_argument('--lambda1', default=5000, type=int, # labeled dataset ratio?
                        help='the weight of classification loss term')
    '''lambda2 -> beta'''
    parser.add_argument('--lambda2', default=0.1, type=int, 
                        help='the weight of beta penalty term, initial value of beta')
    parser.add_argument('--rampup_epoch', default=50, type=int, 
                        help='the max epoch to adjust unsupervised weight')
    # parser.add_argument('--rampdown_epoch', default=50, type=int, 
    #                     help='the last epoch to adjust learning rate')
    parser.add_argument('--entropy_loss', default=False, type=bool,
                        help="add entropy minimization regularization to loss")
    
    '''Optimizer Parameters'''
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('-ad', "--adjust_lr", default=[250, 350], type=arg_as_list, # classifier optimizer scheduling
                        help="The milestone list for adjust learning rate")
    parser.add_argument('--lr_gamma', default=0.5, type=float)
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
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
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
    # args = vars(parser.parse_args(args=['--config_path', 'configs/cifar10_4000.yaml']))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    log_path = f'logs/{args["dataset"]}_{args["labeled_examples"]}'

    datasetL, datasetU, val_dataset, test_dataset, num_classes = fetch_dataset(args, log_path)
    total_length = sum(1 for _ in datasetU)
    
    model = MixtureVAE(args,
                    num_classes,
                    latent_dim=args['latent_dim'],
                    dropratio=args['drop_rate'])
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    
    buffer_model = MixtureVAE(args,
                    num_classes,
                    latent_dim=args['latent_dim'],
                    dropratio=args['drop_rate'])
    buffer_model.build(input_shape=(None, 32, 32, 3))
    buffer_model.set_weights(model.get_weights()) # weight initialization
    
    # '''optimizer'''
    # optimizer = K.optimizers.Adam(learning_rate=args['lr'])
    '''Gradient Cetralized optimizer'''
    class GCAdam(K.optimizers.Adam):
        def get_gradients(self, loss, params):
            grads = []
            gradients = super().get_gradients()
            for grad in gradients:
                grad_len = len(grad.shape)
                if grad_len > 1:
                    axis = list(range(grad_len - 1))
                    grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
                grads.append(grad)
            return grads
    optimizer = GCAdam(learning_rate=args['lr'])
    optimizer_classifier = GCAdam(learning_rate=args['lr'])
    
    train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
    val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
    test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

    test_accuracy_print = 0.
    
    '''prior design'''
    prior_means = np.zeros((num_classes, args['latent_dim']))
    prior_means[:, :num_classes] = np.eye(num_classes) * args['dist']
    prior_means = tf.cast(prior_means[np.newaxis, :, :], tf.float32)

    sigma_vector = np.ones((1, args['latent_dim'])) 
    sigma_vector[0, :num_classes] = args['sigma1']
    sigma_vector[0, num_classes:] = args['sigma2']
    sigma_vector = tf.cast(sigma_vector, tf.float32)
    
    '''initialize beta'''
    beta = tf.cast(args['lambda2'], tf.float32) 
    
    for epoch in range(args['start_epoch'], args['epochs']):
        
        '''learning rate schedule'''
        if epoch == 0:
            '''warm-up'''
            optimizer.lr = args['lr'] * 0.2
        # elif epoch < args['adjust_lr'][0]:
        #     optimizer.lr = args['lr']
        # elif epoch < args['adjust_lr'][1]:
        #     optimizer.lr = args['lr'] * args['lr_gamma']
        # else:
        #     optimizer.lr = args['lr'] * (args['lr_gamma'] ** 2)
            
        '''classifier: learning rate schedule'''
        if epoch == 0:
            '''warm-up'''
            optimizer_classifier.lr = args['lr'] * 0.2
        elif epoch < args['adjust_lr'][0]:
            optimizer_classifier.lr = args['lr']
        elif epoch < args['adjust_lr'][1]:
            optimizer_classifier.lr = args['lr'] * args['lr_gamma']
        else:
            optimizer_classifier.lr = args['lr'] * (args['lr_gamma'] ** 2)
            
        # if epoch >= args['rampdown_epoch']:
        #     optimizer_classifier.lr = args['lr'] * tf.math.exp(-5 * (1. - (args['epochs'] - epoch) / args['epochs']) ** 2)
        #     optimizer_classifier.beta_1 = 0.5
        # # ramp-up
        # optimizer_classifier.lr = weight_schedule(epoch, args['rampup_epoch'], args['lr'])
        # # ramp-down
        # if epoch >= args['epochs'] - args['rampdown_epoch']:
        #     optimizer_classifier.lr = args['lr'] * tf.math.exp(-12.5 * (1. - (args['epochs'] - epoch) / args['rampdown_epoch']) ** 2)
        #     optimizer_classifier.beta_1 = 0.5
            
        if epoch % args['reconstruct_freq'] == 0:
            loss, recon_loss, kl1_loss, kl2_loss, label_mixup_loss, unlabel_mixup_loss, unlabel_ent_loss, accuracy, sample_recon = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, prior_means, sigma_vector, num_classes, total_length, test_accuracy_print)
        else:
            loss, recon_loss, kl1_loss, kl2_loss, label_mixup_loss, unlabel_mixup_loss, unlabel_ent_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, prior_means, sigma_vector, num_classes, total_length, test_accuracy_print)
        # loss, recon_loss, info_loss, nf_loss, accuracy = train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_nf, epoch, args, num_classes, total_length)
        val_recon_loss, val_kl1_loss, val_kl2_loss, val_elbo_loss, val_accuracy = validate(val_dataset, model, epoch, args, prior_means, sigma_vector, num_classes, split='Validation')
        test_recon_loss, test_kl1_loss, test_kl2_loss, test_elbo_loss, test_accuracy = validate(test_dataset, model, epoch, args, prior_means, sigma_vector, num_classes, split='Test')
        
        with train_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
            tf.summary.scalar('recon_loss', recon_loss.result(), step=epoch)
            tf.summary.scalar('kl1', kl1_loss.result(), step=epoch)
            tf.summary.scalar('kl2', kl2_loss.result(), step=epoch)
            tf.summary.scalar('label_mixup_loss', label_mixup_loss.result(), step=epoch)
            tf.summary.scalar('unlabel_mixup_loss', unlabel_mixup_loss.result(), step=epoch)
            tf.summary.scalar('unlabel_ent_loss', unlabel_ent_loss.result(), step=epoch)
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
        label_mixup_loss.reset_states()
        unlabel_mixup_loss.reset_states()
        unlabel_ent_loss.reset_states()
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
        
        if args['beta_trainable']:
            '''beta update'''
            beta = update_beta(model, datasetU, args, total_length)
        
        if epoch == 0:
            optimizer.lr = args['lr']

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
def train(datasetL, datasetU, model, buffer_model, optimizer, optimizer_classifier, epoch, args, beta, prior_means, sigma_vector, num_classes, total_length, test_accuracy_print):
    loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()
    kl1_loss_avg = tf.keras.metrics.Mean()
    kl2_loss_avg = tf.keras.metrics.Mean()
    label_mixup_loss_avg = tf.keras.metrics.Mean()
    unlabel_mixup_loss_avg = tf.keras.metrics.Mean()
    unlabel_ent_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    '''supervised classification weight'''
    lambda1 = tf.cast(args['lambda1'], tf.float32)
    '''un-supervised reconstruction weight'''
    '''unlabel_lambda1 -> lambda2(t)'''
    unlabel_lambda1 = weight_schedule(epoch, args['rampup_epoch'], lambda1)
    # unlabel_lambda1 = weight_schedule(epoch, args['rampup_epoch'], lambda1 * 100. * (args['labeled_examples'] / total_length))
    '''mutual information bound'''
    kl_y_threshold = tf.convert_to_tensor(args['kl_y_threshold'], tf.float32)

    autotune = tf.data.AUTOTUNE
    shuffle_and_batchL = lambda dataset: dataset.shuffle(buffer_size=int(1e5)).batch(batch_size=32, 
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
                                                prior_means, sigma_vector, num_classes, args)
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
            
            elbo = recon_loss + beta * (tf.math.abs(kl1 - kl_y_threshold) + kl2 + cce)
            if args['entropy_loss']:
                '''entropy minimization'''
                probU_aug = model.classify(imageU_aug)
                entropyU = - tf.reduce_sum(tf.reduce_sum(tf.multiply(probU_aug, tf.math.log(tf.clip_by_value(probU_aug, 1e-10, 1.))), axis=-1))
                
                loss = elbo + lambda1 * (cce + mixup_yL + entropyU) + unlabel_lambda1 * mixup_yU
            else:
                entropyU = 0.
                loss = elbo + lambda1 * (cce + mixup_yL) + unlabel_lambda1 * mixup_yU
            
            
        grads = tape.gradient(loss, model.decoder.trainable_variables + model.encoder.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.decoder.trainable_variables + model.encoder.trainable_variables)) 
        # classifier
        grads = tape.gradient(loss, model.classifier.trainable_variables) 
        optimizer_classifier.apply_gradients(zip(grads, model.classifier.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model.classifier, buffer_model.classifier, decay_rate=args['wd'] * optimizer_classifier.lr)
        
        loss_avg(loss)
        recon_loss_avg(recon_loss / args['batch_size'])
        kl1_loss_avg(kl1 / args['batch_size'])
        kl2_loss_avg(kl2 / args['batch_size'])
        label_mixup_loss_avg(mixup_yL / args['batch_size'])
        unlabel_mixup_loss_avg(mixup_yU / args['batch_size'])
        unlabel_ent_loss_avg(entropyU)
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
            'Ent(U)': f'{unlabel_ent_loss_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}',
            'Test Accuracy': f'{test_accuracy_print:.3%}',
            'beta': f'{beta:.4f}'
        })
    
    if epoch % args['reconstruct_freq'] == 0:
        sample_recon = generate_and_save_images1(model, imageU[0][tf.newaxis, ...], num_classes)
        generate_and_save_images2(model, imageU[0][tf.newaxis, ...], num_classes, epoch, f'logs/{args["dataset"]}_{args["labeled_examples"]}/{current_time}')
        return loss_avg, recon_loss_avg, kl1_loss_avg, kl2_loss_avg, label_mixup_loss_avg, unlabel_mixup_loss_avg, unlabel_ent_loss_avg, accuracy, sample_recon
    else:
        return loss_avg, recon_loss_avg, kl1_loss_avg, kl2_loss_avg, label_mixup_loss_avg, unlabel_mixup_loss_avg, unlabel_ent_loss_avg, accuracy
#%%
def validate(dataset, model, epoch, args, prior_means, sigma_vector, num_classes, split):
    nf_loss_avg = tf.keras.metrics.Mean()
    recon_loss_avg = tf.keras.metrics.Mean()   
    kl1_loss_avg = tf.keras.metrics.Mean()   
    kl2_loss_avg = tf.keras.metrics.Mean()   
    elbo_loss_avg = tf.keras.metrics.Mean()   
    accuracy = tf.keras.metrics.Accuracy()
    
    beta = 1.

    dataset = dataset.batch(args['batch_size'], drop_remainder=False)
    for image, label in dataset:
        mean, logvar, prob, y, z, z_tilde, xhat = model(image, training=False)
        recon_loss, kl1, kl2 = ELBO_criterion(prob, xhat, image, mean, logvar, 
                                            prior_means, sigma_vector, num_classes, args)
        cce = - tf.reduce_sum(tf.reduce_sum(tf.multiply(label, tf.math.log(tf.clip_by_value(prob, 1e-10, 1.))), axis=-1))
        
        recon_loss_avg(recon_loss / args['batch_size'])
        kl1_loss_avg(kl1 / args['batch_size'])
        kl2_loss_avg(kl2 / args['batch_size'])
        elbo_loss_avg((recon_loss + beta * (kl1 + kl2 + cce)) / args['batch_size'])
        accuracy(tf.argmax(prob, axis=1, output_type=tf.int32), 
                 tf.argmax(label, axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} ELBO Loss: {elbo_loss_avg.result():.4f}, Recon: {recon_loss_avg.result():.4f}, KL1: {kl1_loss_avg.result():.4f}, KL2: {kl2_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    
    return recon_loss_avg, kl1_loss_avg, kl2_loss_avg, elbo_loss_avg, accuracy
#%%
def weight_schedule(epoch, epochs, weight_max):
    return weight_max * tf.math.exp(-5. * (1. - min(1., epoch/epochs)) ** 2)
#%%
def update_beta(model, datasetU, args, total_length):
    autotune = tf.data.AUTOTUNE
    batch = lambda dataset: dataset.batch(batch_size=args['batch_size'], drop_remainder=False).prefetch(autotune)
    iteratorU = iter(batch(datasetU))
    
    iteration = total_length // args['batch_size'] 
    beta = 0.
    for _ in range(iteration + 1):
        imageU, _ = next(iteratorU)
        xhatU = model(imageU)[-1]
        beta += tf.reduce_sum(tf.reduce_mean(tf.math.square(xhatU - imageU), axis=-1))
    beta = beta / 50000. + tf.cast(args['lambda2'], tf.float32)
    return beta
#%%
if __name__ == '__main__':
    main()
#%%