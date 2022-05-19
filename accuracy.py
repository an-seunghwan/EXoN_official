#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorboard as tb
import numpy as np
#%%
"""classification error"""
with open("accuracy.txt", "w") as file:
    
    for beta in [0.01, 0.05, 0.1, 0.5, 1]:
        # dir = '/Users/anseunghwan/Documents/GitHub/EXoN_official/logs/cifar10_4000/repeated(beta_{})'.format(beta)
        dir = r'D:\EXoN_official\logs\cifar10_4000\repeated(beta_{})'.format(beta)
        file_list = [x for x in os.listdir(dir) if x not in ['.DS_Store', 'datasets', 'etc']]
        
        error = []
        for i in range(len(file_list)):
            path = dir + '/{}/test'.format(file_list[i])
            event_acc = EventAccumulator(path)
            event_acc.Reload()
            tag = 'accuracy'
            event_list = event_acc.Tensors(tag)
            value = tf.io.decode_raw(event_list[-1].tensor_proto.tensor_content, 
                                    event_list[-1].tensor_proto.dtype)
            error.append(100. * (1. - value.numpy()[0]))

        file.write("{} | mean: {:.3f}, std: {:.3f}\n\n".format(beta, np.mean(error), np.std(error)))
#%%
"""inception score"""
#%%