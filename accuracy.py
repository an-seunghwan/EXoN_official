#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import re
#%%
"""Table 1 and 2"""
with open("accuracy.txt", "w") as file:
    for beta in [0.01, 0.05, 0.1, 0.5, 1]:
        dir = '/Users/anseunghwan/Documents/GitHub/EXoN_official/logs/cifar10_4000/repeated(beta_{})'.format(beta)
        model_list = [d for d in os.listdir(dir) if d != '.DS_Store']
        
        inception = []
        card = []
        error = []
        for i in range(len(model_list)):
            with open(dir + '/' + model_list[i] + '/result.txt', 'r') as f:
                result = f.readlines()
            result = ' '.join(result) 
            
            """Inception Score"""
            idx1 = re.search(' mean: ', result).span()[1]
            idx2 = re.search(', std: ', result).span()[0]
            inception.append(float(result[idx1:idx2]))
            
            """cardinality"""
            idx1 = re.search('cardinality of activated latent subspace: ', result).span()[1]
            idx2 = re.search('inception', result).span()[0]
            card.append(float(result[idx1:idx2-4]))
            
            """test classification error"""
            idx1 = re.search('TEST classification error: ', result).span()[1]
            idx2 = re.search('%', result).span()[0]
            error.append(float(result[idx1:idx2]))
            
        file.write("beta: {} | Inception Score | mean: {:.3f}, std: {:.3f}\n".format(beta, np.mean(inception), np.std(inception)))
        file.write("beta: {} | cardinality | mean: {:.3f}, std: {:.3f}\n".format(beta, np.mean(card), np.std(card)))
        file.write("beta: {} | test classification error | mean: {:.3f}, std: {:.3f}\n\n".format(beta, np.mean(error), np.std(error)))
#%%