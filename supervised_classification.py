# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:17:35 2021

@author: kajul
"""

import numpy as np
import json
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

import pandas as pd
import os
import matplotlib.pyplot as plt

experiment_dir = "ClassESOF1"
epoch = "1000"
latent_size = 64

from networks.deep_sdf_classifier import Classifier

#%% Create dataloaders
class LatentESOFDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pkl_file, filenames, latent_path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pkl_frame = pd.read_pickle(pkl_file)
        self.filenames = filenames
        self.latent_path = latent_path

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        full_path = os.path.join(self.latent_path,self.filenames[idx] + ".pth")
        latent_vector = torch.load(full_path).cpu().detach().numpy()[0,0,:]
        gender = int(self.pkl_frame.loc[self.filenames[idx]]["Gender"])
        
        #sample = {'latent': latent_vector, 'gender': gender}

        return latent_vector, gender   
    
#%% Load network
classification_net = Classifier(latent_size)   
data = torch.load("H:/DeepSDF-master/experiments/"+experiment_dir+"/ModelParameters/"+epoch+"_classifier.pth")
classification_net.load_state_dict(data["model_state_dict"])

criterion = nn.NLLLoss()
    
#%% TEST
pkl_file = 'H:/ESOF/ESOF_dataframe.pkl'
latent_path = 'H:/DeepSDF-master/experiments/'+experiment_dir+'/Reconstructions/'+epoch+'/Codes/ESOF_NDF/All/'
with open('H:/DeepSDF-master/examples/splits/ESOF_NDF_test.json') as json_file:
    split = json.load(json_file)        
test_dataset = LatentESOFDataset(pkl_file,split["ESOF_NDF"]["All"],latent_path)
test_loader = DataLoader(test_dataset,batch_size=10,shuffle=True)

        
# run a test loop
test_loss = 0
correct = 0
for data, target in test_loader:    
    data, target = Variable(data, volatile=True), Variable(target)
    net_out = classification_net(data.cuda())
    # sum up batch loss
    test_loss += criterion(net_out, target.cuda()).item()
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.cuda().data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))  

#%% From saved latent codes
data = torch.load('H:/DeepSDF-master/experiments/'+experiment_dir+'/LatentCodes/'+epoch+".pth")['latent_codes']['weight']
net_out = classification_net(data.cuda())
pred = np.array(net_out.data.max(1)[1].cpu())

nESOF = data.shape[0]
df = pd.read_pickle('H:/ESOF/ESOF_dataframe.pkl')
with open('H:/DeepSDF-master/examples/splits/ESOF_NDF_train.json') as json_file:
    split = json.load(json_file)    
gender_marker = np.zeros(nESOF)
for i in range(nESOF): 
    fileid = split["ESOF_NDF"]["All"][i]
    try: 
        if df.loc[fileid]["Gender"] == 1: 
            gender_marker[i] = 1
    except: 
        print("Not found: ", fileid)

pca = PCA(n_components = latent_size)
pca.fit(data)
pca_data = pca.transform(data)

x = pca_data[:,0]
y = pca_data[:,1]

plt.figure()
plt.scatter(x[pred==0],y[pred==0],c='r')
plt.scatter(x[pred==1],y[pred==1],c='b')

plt.figure()
plt.scatter(x[gender_marker==0],y[gender_marker==0],c='r')
plt.scatter(x[gender_marker==1],y[gender_marker==1],c='b')

## Difference
plt.figure()
plt.scatter(x[gender_marker==0],y[gender_marker==0],c='r')
plt.scatter(x[gender_marker==1],y[gender_marker==1],c='b')
plt.scatter(x[gender_marker!=pred],y[gender_marker!=pred],c='k')

np.sum(gender_marker == pred)


#%% 
data = torch.load('H:/DeepSDF-master/experiments/'+experiment_dir+'/LatentCodes/'+epoch+".pth")['latent_codes']['weight']
n_train = data.shape[0]
n_latent = data.shape[1]
with open('H:/DeepSDF-master/examples/splits/ESOF_NDF_train.json') as json_file:
    split = json.load(json_file)  
train_latent = np.zeros((n_train,n_latent))
for i in range(n_train):
    fileid = split["ESOF_NDF"]["All"][i]
    rec_latent = torch.load('H:/DeepSDF-master/experiments/'+experiment_dir+'/Reconstructions/'+epoch+'/Codes/ESOF_NDF/All/'+fileid+'.pth').cpu()
    train_latent[i,:] = rec_latent.detach().numpy()[0,0,:]

pca = PCA(n_components = latent_size)
pca.fit(data)
pca_data = pca.transform(data)
pca_recon = pca.transform(train_latent)

plt.figure()
for i in range(data.shape[0]):
    opt_latent = pca_data[i,:]
    recon_latent = pca_recon[i,:]
    
    plt.scatter(opt_latent[0],opt_latent[1],c='k')
    plt.scatter(recon_latent[0],recon_latent[1],c='r')

plt.figure()
for i in range(data.shape[0]):
    opt_latent = pca_data[i,:]
    recon_latent = pca_recon[i,:]
    
    plt.plot([opt_latent[0], recon_latent[0]],[opt_latent[1], recon_latent[1]],'-')
    
    plt.scatter(opt_latent[0],opt_latent[1],c='k')
    plt.scatter(recon_latent[0],recon_latent[1],c='r')
