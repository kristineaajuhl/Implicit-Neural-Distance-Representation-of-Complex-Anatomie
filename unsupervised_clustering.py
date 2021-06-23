# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:43:18 2021

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

experiment_dir = "3data_oversample"
epoch = "1000"

from sklearn.cluster import KMeans

#%% Load latent vectors

with open('H:/DeepSDF-master/examples/splits/3data_NDF_train_oversample.json') as json_file:
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

#%%

kmeans_kwargs = { "init": "random", "n_init": 10, "max_iter": 300,"random_state": 42}

plt.figure()
sse = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(latent)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse,'-')
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


nc = 6
kmeans = KMeans(n_clusters=nc, **kmeans_kwargs)
pred = kmeans.fit_predict(latent)
pca = PCA(n_components = n_latent)
pca.fit(latent)
latent_pca = pca.transform(latent)
x = latent_pca[:,0]
y = latent_pca[:,1]

# gt = np.hstack((1*np.ones(nESOF), 2*np.ones(nEARS), 0*np.ones(nLA)))
# pred2 = pred.copy()
# #pred2[pred2 == 2] = 1
# pred2[pred2 == 3] = 2
# np.sum(gt==pred2)

# predEARS = pred[nESOF:nESOF+nEARS]
# predEARS[predEARS == 3] = 0
# predEARS[predEARS == 2] = 1
# np.sum(predEARS==ear_side) / len(ear_side)

plt.figure()
#colors = ["Blue","Green","Red"]
#colors = ["Blue","Red","lightgreen","green"]
colors = ["Blue","Red","lightgreen","green","cyan","magenta"]
for i in range(nc):
    plt.scatter(x[pred==i],y[pred==i],c=colors[i])
    #plt.scatter(x[pred==gt],y[pred==gt],c='k')
plt.xlabel("PCA1")
plt.ylabel("PCA2")

#%%

Xr = np.arange(1,20)
plt.figure()
for r in Xr: 
    latentX = latent_pca[:,0:r]

    # Approximate same scale - no rason to scale the features
    kmeans_kwargs = { "init": "random", "n_init": 10, "max_iter": 300,"random_state": 42}
      # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
          kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
          kmeans.fit(latentX)
          sse.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), sse,'-')

plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.legend(Xr)
plt.show()