#%%
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy.io as sio
import urllib.request as urllib
import cv2
from tqdm import tqdm
#%%
# def download_dataset(save_path, verbose=True):
#     if(os.path.isfile(save_path + "train_32x32.mat") == False):
#         if(verbose): print("Downloading train_32x32.mat...")
#         urllib.urlretrieve ("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", save_path + "train_32x32.mat")
#         if(verbose): print("Done!")
#     if(os.path.isfile(save_path + "test_32x32.mat") == False):
#         if(verbose): print("Downloading test_32x32.mat...")
#         urllib.urlretrieve ("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", save_path + "test_32x32.mat")
#         if(verbose): print("Done!")
            
# download_dataset(save_path=save_path)
#%%
def download_dataset(dataset_name, save_path):
    assert dataset_name == 'svhn'
    
    train_mat = sio.loadmat(save_path + 'train_32x32.mat')
    train_y = train_mat["y"]
    test_mat = sio.loadmat(save_path + 'test_32x32.mat')
    test_y = test_mat["y"]

    train_x = train_mat["X"]
    train_x = np.swapaxes(train_x, 2, 3) #[32, 32, 3, None] > [32, 32, None, 3]
    train_x = np.swapaxes(train_x, 1, 2) #[32, 32, None, 3] > [32, None, 32, 3]
    train_x = np.swapaxes(train_x, 0, 1) #[32, None, 32, 3] > [None, 32, 32, 3]
    train_x = (train_x - 127.5) / 127.5
    
    test_x = test_mat["X"]
    test_x = np.swapaxes(test_x, 2, 3) #[32, 32, 3, None] > [32, 32, None, 3]
    test_x = np.swapaxes(test_x, 1, 2) #[32, 32, None, 3] > [32, None, 32, 3]
    test_x = np.swapaxes(test_x, 0, 1) #[32, None, 32, 3] > [None, 32, 32, 3]
    test_x = (test_x - 127.5) / 127.5
    
    return train_x, train_y, test_x, test_y
#%%
'''
- labeled and validation are disjoint
- unlabeled includes validation
- unlabeled includes labeled
'''
def split_dataset(train_x, train_y, num_labeled, num_validations, num_classes, args):
    np.random.seed(args['seed'])
    idx = np.random.choice(range(len(train_x)), len(train_x), replace=False)
    train_x = train_x[idx]
    train_y = train_y[idx]
    counter = [0 for _ in range(num_classes)]
    labeled = []
    unlabeled = []
    validation = []
    for x, y in tqdm(zip(train_x, train_y), desc='split_dataset'):
        label = y[0] % 10
        counter[label] += 1
        if counter[label] <= (num_validations / num_classes):
            validation.append({
                'image': x,
                'label': label
            })
        elif counter[label] <= (num_validations / num_classes + num_labeled / num_classes):
            labeled.append({
                'image': x,
                'label': label
            })
        unlabeled.append({
            'image': x,
            'label': tf.convert_to_tensor(-1, dtype=tf.int64)
        })
    labeled = _list_to_tf_dataset(labeled)
    unlabeled = _list_to_tf_dataset(unlabeled)
    validation = _list_to_tf_dataset(validation)
    return labeled, unlabeled, validation
#%%
def _list_to_tf_dataset(dataset):
    def _dataset_gen():
        for example in dataset:
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image':tf.float32, 'label':tf.int64},
        output_shapes={'image': (32, 32, 3), 'label': ()}
    )
#%%
def serialize_example(example, num_classes):
    image = example['image']
    label = example['label']
    image = image.astype(np.float32).tobytes()
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
# dataset_name = 'svhn'
# args = {
#     'seed': 0,
#     'labeled_examples': 1000, 
#     'validation_examples': 1000,
#     'num_classes': 10
# }
def fetch_dataset(dataset_name, save_path, args):
    num_classes = 10
    
    if any([not os.path.exists(f'{save_path}/{split}.tfrecord') for split in ['trainL', 'trainU', 'validation', 'test']]):
        train_x, train_y, test_x, test_y = download_dataset(dataset_name, save_path)
        
        trainL, trainU, validation = split_dataset(
            train_x, train_y,
            num_labeled=args['labeled_examples'],
            num_validations=args['validation_examples'],
            num_classes=num_classes,
            args=args)
        
        for name, dataset in [('trainL', trainL), ('trainU', trainU), ('validation', validation)]:
            writer = tf.io.TFRecordWriter(f'{save_path}/{name}.tfrecord'.encode('utf-8'))
            for x in tfds.as_numpy(dataset):
                example = serialize_example(x, num_classes)
                writer.write(example)
        
        name = 'test'
        writer = tf.io.TFRecordWriter(f'{save_path}/{name}.tfrecord'.encode('utf-8'))
        test_ = []
        for x, y in zip(test_x, test_y):
            test_.append({
                    'image': x,
                    'label': y[0] % 10
                })
        test_dataset = _list_to_tf_dataset(test_)
        for x in tfds.as_numpy(test_dataset):
            example = serialize_example(x, num_classes=num_classes)
            writer.write(example)
    
    trainL = tf.data.TFRecordDataset(f'{save_path}/trainL.tfrecord'.encode('utf-8')).map(deserialize_example)
    trainU = tf.data.TFRecordDataset(f'{save_path}/trainU.tfrecord'.encode('utf-8')).map(deserialize_example)
    validation = tf.data.TFRecordDataset(f'{save_path}/validation.tfrecord'.encode('utf-8')).map(deserialize_example)
    test = tf.data.TFRecordDataset(f'{save_path}/test.tfrecord'.encode('utf-8')).map(deserialize_example)
    
    return trainL, trainU, validation, test, num_classes
#%%
# autotune = tf.data.AUTOTUNE
# shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e3)).batch(
#     batch_size=8, 
#     drop_remainder=False).prefetch(autotune)
# iteratorL = iter(shuffle_and_batch(test))
# x, y = next(iteratorL)
# print(x.shape)
# print(y.shape)
#%%