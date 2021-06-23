# Implicit-Neural-Distance-Representation-of-Complex-Anatomie
Repository linked to publication: Implicit Neural Distance Representation for Unsupervised and Supervised Classification of Complex Anatomies (Accepted for MICCAI2021).

Authors: Kristine Aavild Juhl, Xabier Morales, Oscar Camara, Ole de Backer and Rasmus Reinhold Paulsen

The code is adapted from deepSDF (https://github.com/facebookresearch/DeepSDF)

# Prepare training data
The preprocessing is a fully pythonic pipeline taking .vtk files as input and outputting arrays of point-coordinates and distances to be used for training. 
The meshes can be alligned using a set of anatomical landmarks ('data_generator/make_train_data_LM.py') or using iterative closest point ('data_generator/make_train_data.py')
The following changes should be made in the respective files to preprocess your own dataset:
- l. 17: True/False indicates if the data should be prealigned
- l. 21: A list of fileids should be saved in a .txt file and the filepath should be specified.
- l. 22: A fileid should be given as the template for rigid alignemnt 
- l. 43-47: Specify filepaths for the processed data
- l. 135+151 / 110+126: Specify the scale_factor so that the largest mesh fits the unit-sphere (number can be found by running l. 133 after aligning all shapes)
- l. 178+179 / 153+154: Set standard deviation on noise pertubation of sampling points

To visualize the mesh and the sampled points run 'check_trainingdata'. 

# Train autodecoder model



