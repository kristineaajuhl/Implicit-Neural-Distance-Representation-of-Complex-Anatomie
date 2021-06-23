# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 08:43:38 2020

@author: kajul
"""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import json
from skimage.transform import rescale
import torch
from sklearn.manifold import TSNE
import matplotlib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

experiment_dir = "3data"
epoch = "1000"

#%% Load latent vectors

with open('H:/DeepSDF-master/examples/splits/3data_NDF_train.json') as json_file:
    split = json.load(json_file)
nESOF = len(split["ESOF_NDF"]["All"])
nEARS = len(split["EARS_NDF"]["All"])
nLA = len(split["LA_NDF"]["All"])
n_latent = 128

latent = np.zeros((nESOF+nEARS+nLA,n_latent))
base_dir = 'H:/DeepSDF-master/experiments/'+experiment_dir+'/Reconstructions/'+epoch+'/Codes/'
for i in range(nESOF):
    filename = base_dir + "/ESOF_NDF/All/" + split["ESOF_NDF"]["All"][i] + ".pth"
    rec_latent = torch.load(filename).cpu()
    latent[i,:] = rec_latent.detach().numpy()[0,0,:]
index_end = i+1
for k in range(nEARS):
    filename = base_dir + "/EARS_NDF/All/" + split["EARS_NDF"]["All"][k] + ".pth"
    rec_latent = torch.load(filename).cpu()
    latent[index_end + k,:] = rec_latent.detach().numpy()[0,0,:]
index_end += k+1
for j in range(nLA):
    filename = base_dir + "/LA_NDF/All/" + split["LA_NDF"]["All"][j] + ".pth"
    rec_latent = torch.load(filename).cpu()
    latent[index_end + j,:] = rec_latent.detach().numpy()[0,0,:]

# EARS side and ESOF gender:
ear_side = np.zeros(nEARS)
for k in range(nEARS):
    side = split["EARS_NDF"]["All"][k][7]
    if side == "R": 
        ear_side[k] = 1
        
df = pd.read_pickle('H:/ESOF/ESOF_dataframe.pkl')
gender_marker = np.zeros(nESOF)
marker_age_size = np.zeros(nESOF)
for i in range(nESOF): 
    fileid = split["ESOF_NDF"]["All"][i]
    try: 
        if df.loc[fileid]["Gender"] == 1: 
            gender_marker[i] = 1
        marker_age_size[i] = df.loc[fileid]["Age"]
    except: 
        print("Not found: ", fileid)

#%%tSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000)
tsne_results = tsne.fit_transform(np.vstack((latent)))
    
#plt.scatter(tsne_results[:,0],tsne_results[:,1])
#np.save('H:/DeepSDF-master/experiments/'+experiment_dir+'/Reconstructions/'+epoch+'/Codes/tSNE.npy',tsne_results)
#tsne_results = np.load('H:/DeepSDF-master/experiments/'+experiment_dir+'/Reconstructions/'+epoch+'/Codes/tSNE.npy')


x = tsne_results[:,0]
y = tsne_results[:,1]

# fig, ax = plt.subplots()
# plt.scatter(x[0:nESOF],y[0:nESOF],c='r')
# plt.scatter(x[nESOF:nESOF+nEARS],y[nESOF:nESOF+nEARS],c='g')
# plt.scatter(x[nEARS+nESOF::],y[nEARS+nESOF::],c='b')
#plt.legend(["ESOF","EARS","LA"])

plt.scatter(x[0:nESOF][gender_marker==1],y[0:nESOF][gender_marker==1],c='r', marker='o')
plt.scatter(x[0:nESOF][gender_marker==0],y[0:nESOF][gender_marker==0],c='r', marker='x')
plt.scatter(x[nESOF:nESOF+nEARS][ear_side==1],y[nESOF:nESOF+nEARS][ear_side==1],c='g',marker='o')
plt.scatter(x[nESOF:nESOF+nEARS][ear_side==0],y[nESOF:nESOF+nEARS][ear_side==0],c='g',marker='x')
plt.scatter(x[nEARS+nESOF::],y[nEARS+nESOF::],c='b')

plt.xlabel("tSNE1")
plt.ylabel("tSNE2")
plt.legend(["ESOF Male", "ESOF Female","EARS Right", "EARS_Left","LA"])


#%% PCA
plt.figure()
pca = PCA(n_components = n_latent)
pca.fit(latent)
plt.plot(pca.explained_variance_ratio_, '-*k')

X = latent.copy().T
randomX = np.empty(X.shape)
for i in range(n_latent):
    col = X[:,i].copy()
    np.random.shuffle(col)
    randomX[:,i] = col.copy()

pca2 = PCA(n_components = n_latent)
pca2.fit(randomX.T)
plt.plot(pca2.explained_variance_ratio_, '-*r')

plt.xlabel("Mode")
plt.ylabel("Explained Variance")
plt.legend(["Real","Random"])

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
#%% PCA

latent_pca = pca.transform(latent)
x = latent_pca[:,0]
y = latent_pca[:,1]

plt.figure()
plt.scatter(x[0:nESOF],y[0:nESOF],c='r')
plt.scatter(x[nESOF:nESOF+nEARS],y[nESOF:nESOF+nEARS],c='g')
plt.scatter(x[nEARS+nESOF::],y[nEARS+nESOF::],c='b')

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(["ESOF","EARS","LA"])

plt.figure()        
plt.scatter(x[0:nESOF][gender_marker==1],y[0:nESOF][gender_marker==1],c='r', marker='o')
plt.scatter(x[0:nESOF][gender_marker==0],y[0:nESOF][gender_marker==0],c='r', marker='x')
plt.scatter(x[nESOF:nESOF+nEARS][ear_side==1],y[nESOF:nESOF+nEARS][ear_side==1],c='g',marker='o')
plt.scatter(x[nESOF:nESOF+nEARS][ear_side==0],y[nESOF:nESOF+nEARS][ear_side==0],c='g',marker='x')
plt.scatter(x[nEARS+nESOF::],y[nEARS+nESOF::],c='b')

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(["ESOF Male", "ESOF Female","EARS Right", "EARS_Left","LA"])


fig, ax = plt.subplots()
ax.scatter(x[nESOF:nESOF+nEARS][ear_side==1],y[nESOF:nESOF+nEARS][ear_side==1],marker='o')
ax.scatter(x[nESOF:nESOF+nEARS][ear_side==0],y[nESOF:nESOF+nEARS][ear_side==0],marker='x')
for i, txt in enumerate(split["EARS_NDF"]["All"]):
    if np.remainder(i,1) == 0:
        ax.annotate(txt,(x[nESOF:nESOF+nEARS][i],y[nESOF:nESOF+nEARS][i]))

# Many modes
# fig, axs = plt.subplots(7,7)
# axs = axs.ravel()
# i = 0
# for i1 in range(7):   
#     x = latent_pca[:,i1]
#     for i2 in range(7):
#         y = latent_pca[:,i2]

#         axs[i].scatter(x[0:nESOF],y[0:nESOF],c='r')
#         axs[i].scatter(x[nESOF:nESOF+nEARS],y[nESOF:nESOF+nEARS],c='g')
#         axs[i].scatter(x[nEARS+nESOF::],y[nEARS+nESOF::],c='b')
        
#         i += 1


#%% ESOF-only from the full learned dataset

ESOF_latent = latent[0:nESOF,:]    
pca = PCA(n_components = n_latent)
pca.fit(ESOF_latent)
ESOF_latent_pca = pca.transform(ESOF_latent)
x = ESOF_latent_pca[:,0]
y = ESOF_latent_pca[:,1]
plt.scatter(x[gender_marker==1],y[gender_marker==1],c='b',s=marker_age_size)
plt.scatter(x[gender_marker==0],y[gender_marker==0],c='r',s=marker_age_size)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(["Male","Female"])