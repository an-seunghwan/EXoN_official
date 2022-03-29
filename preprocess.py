#%%
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
from tqdm import tqdm
#%%
def download_dataset(dataset_name):
    train = None
    test = None
    if dataset_name == 'svhn':
        dataset = tfds.load(name='svhn_cropped')
        train = dataset['train']
        test = dataset['test']
    
    elif dataset_name == 'svhn+extra':
        dataset = tfds.load(name='svhn_cropped')
        train = dataset['train']
        train.concatenate(dataset['extra'])
        test = dataset['test']
    
    elif dataset_name == 'cifar10':
        dataset = tfds.load(name='cifar10')
        train = dataset['train']
        test = dataset['test']
    
    elif dataset_name == 'cifar100':
        dataset = tfds.load(name='cifar100')
        train = dataset['train']
        test = dataset['test']
        
    elif dataset_name == 'cmnist':
        dataset = tfds.load(name='mnist')
        train = dataset['train']
        test = dataset['test']
    
    return  train, test
#%%
color_list = [
    (255, 0, 0), # red 
    # (255, 0, 128), # rose 
    (255, 0, 255), # magenta 
    # (128, 0, 255), # violet
    (0, 0, 255), # blue 
    # (0, 128, 255), # azure 
    (0, 255, 255), # cyan 
    # (0, 255, 128), # spring green 
    (0, 255, 0), # green
    # (128, 255, 0), # chartreuse 
    (255, 255, 0), # yellow 
    # (255, 128, 0), # orange
]
#%%
def colored_mnist(image):
    '''
    color list: [
        red, rose, magenta, violet,
        blue, azure, cyan, spring green, green
        chartreuse, yellow, orange
    ]
    '''
    
    image = tf.image.resize(image, [32, 32], method='nearest')
    
    # if tf.random.uniform((1, 1)) > 0.5:
    # color
    image = tf.cast(image, tf.float32) / 255.
    # color = np.random.uniform(0., 1., 3)
    # color = color / np.linalg.norm(color)
    color = np.array(color_list[np.random.choice(range(len(color_list)), 1)[0]]) / 255.
    image = image * color[tf.newaxis, tf.newaxis, :]
    return image
    # else:
    #     # edge detection
    #     image = cv2.Canny(image.numpy(), 10., 255.)
    #     image[np.where(image > 0)] = 1.
    #     image[np.where(image <= 0)] = 0.
    #     # color
    #     # color = np.random.uniform(0., 1., 3)
    #     # color = color / np.linalg.norm(color)
    #     color = np.array(color_list[np.random.choice(range(len(color_list)), 1)[0]]) / 255.
    #     image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
    #     # width
    #     kernel = np.ones((1, 1))
    #     image = cv2.dilate(image, kernel)
    #     return image
#%%
def _list_to_tf_dataset(dataset, args):
    def _dataset_gen():
        for example in dataset:
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image':tf.float32, 'label':tf.int64} if args['dataset'] == 'cmnist' else {'image':tf.uint8, 'label':tf.int64},
        output_shapes={'image': (32, 32, 3), 'label': ()}
        # output_shapes={'image': (args['image_size'], args['image_size'], args['channel']), 'label': ()}
    )
#%%
'''
- labeled and validation are disjoint
- unlabeled includes validation
- unlabeled includes labeled
'''
def split_dataset(dataset, num_labeled, num_validations, num_classes, args):
    np.random.seed(args['seed'])
    dataset = dataset.shuffle(buffer_size=10000, seed=args['seed'])
    counter = [0 for _ in range(num_classes)]
    labeled = []
    unlabeled = []
    validation = []
    for example in tqdm(iter(dataset), desc='split_dataset'):
        label = int(example['label'])
        counter[label] += 1
        if args['dataset'] == 'cmnist':
            if counter[label] <= (num_validations / num_classes):
                validation.append({
                    'image': colored_mnist(example['image']),
                    'label': example['label']
                })
            elif counter[label] <= (num_validations / num_classes + num_labeled / num_classes):
                labeled.append({
                    'image': colored_mnist(example['image']),
                    'label': example['label']
                })
            unlabeled.append({
                'image': colored_mnist(example['image']),
                'label': tf.convert_to_tensor(-1, dtype=tf.int64)
            })
        else:
            if counter[label] <= (num_validations / num_classes):
                validation.append({
                    'image': example['image'],
                    'label': example['label']
                })
            elif counter[label] <= (num_validations / num_classes + num_labeled / num_classes):
                labeled.append({
                    'image': example['image'],
                    'label': example['label']
                })
            unlabeled.append({
                'image': example['image'],
                'label': tf.convert_to_tensor(-1, dtype=tf.int64)
            })
    labeled = _list_to_tf_dataset(labeled, args)
    unlabeled = _list_to_tf_dataset(unlabeled, args)
    validation = _list_to_tf_dataset(validation, args)
    return labeled, unlabeled, validation
#%%
def cmnist_test_dataset(dataset, args):
    test = []
    for example in tqdm(iter(dataset), desc='cmnist_test_dataset'):
        test.append({
            'image': colored_mnist(example['image']),
            'label': example['label']
        })
    test = _list_to_tf_dataset(test, args)
    return test
#%%
def normalize_image(image):
    # image = image / 255.
    image = (image - 127.5) / 127.5
    return image
#%%
def serialize_example(example, num_classes, args):
    image = example['image']
    label = example['label']
    if args['dataset'] == 'cmnist':
        image = image.astype(np.float32).tobytes()
    else:
        image = normalize_image(image.astype(np.float32)).tobytes()
    label = np.eye(num_classes).astype(np.float32)[label].tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
    })) 
    return example.SerializeToString()
#%%
def deserialize_example(serialized_string):
    image_feature_description = { 
        'image': tf.io.FixedLenFeature([], tf.string), 
        'label': tf.io.FixedLenFeature([], tf.string), 
    } 
    example = tf.io.parse_single_example(serialized_string, image_feature_description) 
    image = tf.reshape(tf.io.decode_raw(example["image"], tf.float32), (32, 32, 3))
    label = tf.io.decode_raw(example["label"], tf.float32) 
    return image, label
#%%
def fetch_dataset(args, log_path):
    dataset_path = f'{log_path}/datasets'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    num_classes = 100 if args['dataset'] == 'cifar100' else 10
    
    if any([not os.path.exists(f'{dataset_path}/{split}.tfrecord') for split in ['trainL', 'trainU', 'validation', 'test']]):
        train, test = download_dataset(dataset_name=args['dataset'])
        
        trainL, trainU, validation = split_dataset(dataset=train,
                                                num_labeled=args['labeled_examples'],
                                                num_validations=args['validation_examples'],
                                                num_classes=num_classes,
                                                args=args)
        
        if args['dataset'] == 'cmnist':
            test = cmnist_test_dataset(dataset=test,
                                       args=args)
            
        for name, dataset in [('trainL', trainL), ('trainU', trainU), ('validation', validation), ('test', test)]:
            writer = tf.io.TFRecordWriter(f'{dataset_path}/{name}.tfrecord'.encode('utf-8'))
            for x in tfds.as_numpy(dataset):
                example = serialize_example(x, num_classes, args)
                writer.write(example)
    
    trainL = tf.data.TFRecordDataset(f'{dataset_path}/trainL.tfrecord'.encode('utf-8')).map(deserialize_example)
    trainU = tf.data.TFRecordDataset(f'{dataset_path}/trainU.tfrecord'.encode('utf-8')).map(deserialize_example)
    validation = tf.data.TFRecordDataset(f'{dataset_path}/validation.tfrecord'.encode('utf-8')).map(deserialize_example)
    test = tf.data.TFRecordDataset(f'{dataset_path}/test.tfrecord'.encode('utf-8')).map(deserialize_example)
    
    return trainL, trainU, validation, test, num_classes
#%%