# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:54:52 2021

@author: kajul
"""

import vtk
import os 
import json
from deep_sdf import UDFmesh_methods as Methods

if __name__ == '__main__': 
    n_threads = 12
    mesh_UDF = True
    rotate_surf = True
    eval_reconstruction = True
    
    with open('H:/DeepSDF-master/examples/splits/3data_NDF_train.json') as json_file:
        split = json.load(json_file)
    UDF_path = 'E:/deepSDF-experiments/Reconstructions/1000/UDF/'
    temp_path = 'E:/deepSDF-experiments/Reconstructions/1000/temp/'
    final_path = 'E:/deepSDF-experiments/Reconstructions/1000/Meshes/'
    mrf_exe = "C:\Program Files\MRFSurface\MRFSurface.exe"
    eval_file = 'E:/deepSDF-experiments/Reconstructions/1000/eval_train.txt'
        
    for dataset in split:
        print(dataset)
        if dataset == "ESOF_NDF":
            continue
        if dataset == "LA_NDF":
            continue
        name_list = split[str(dataset)]["All"]
        n_names = len(name_list)
        
        UDF_list =  []
        for name in name_list: 
            UDF_list.append(UDF_path + str(dataset) + "/All/" + name + ".mhd")
        temp_list = []
        for name in name_list: 
            temp_list.append(temp_path + str(dataset) + "/All/" + name + ".vtk")
        final_list = []
        for name in name_list: 
            final_list.append(final_path + str(dataset) + "/All/" + name + ".vtk")
        true_list = []
        for name in name_list: 
            if dataset == "ESOF_NDF":
                true_list.append('E:/DATA/ESOF/mesh_normalized/'+name+'.vtk')
            elif dataset == "EARS_NDF":
                true_list.append('E:/DATA/EARS/mesh_normalized/'+name+'.vtk')
            elif dataset == "LA_NDF":
                true_list.append('E:/DATA/LA/mesh_normalized/'+name+'.vtk')
                
        # Create surface from UDF
        if mesh_UDF: 
            meshArgs = list(zip(UDF_list,temp_list,[mrf_exe]*n_names))
            Methods.imap_unordered_bar(Methods.mesh_UDF, meshArgs, n_threads)
        
        # Flip and rotate mesh from UDF
        if rotate_surf: 
            rotateArgs = list(zip(temp_list,final_list))
            Methods.imap_unordered_bar(Methods.rotate_vtk, rotateArgs, n_threads)
            
        if eval_reconstruction:
            evalArgs = list(zip(final_list, true_list, [eval_file]*n_names))
            Methods.imap_unordered_bar(Methods.eval_reconstruction, evalArgs, n_threads)
    



