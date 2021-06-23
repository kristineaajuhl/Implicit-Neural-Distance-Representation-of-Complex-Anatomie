# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:30:47 2020

@author: kajul
"""

import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy 

import vtk




align = True

#np.random.seed(12345)
#%%
list_of_fileids = 'H:/ESOF/ESOF_ids.txt'
align_to = '20140621103743'
fileids = []
f =  open(list_of_fileids,"r")
for x in f:
    if not x.strip == align_to:
        fileids.append(x.strip())

# rmdir = os.listdir('E:/DATA/ESOF/SDF_samples/')
# for rm in rmdir: 
#     if rm[:-4] in fileids: 
#         fileids.remove(rm[:-4])

# fileids.remove('20140622164708')
fileids.insert(0,align_to)

## Create npz file with xyz-coordinates and correspondeing SDF value
sf = []
for fileid in fileids:
    print("Processing file: ",fileid)
    

    sdf_dir =    'E:/DATA/ESOF/NDF/'+fileid+'/'+fileid+'_DistanceField.nii'
    output_file = 'E:/DATA/ESOF/NDF_samples/'+fileid+'.npz'      
    input_file = 'H:/ESOF/Cut_v2/'+fileid+'_standard_cut.vtk' 
    norm_file = 'E:/DATA/ESOF/mesh_normalized/'+fileid+'.vtk' 
    LM_file = 'M:/ESOF/' + fileid + "/" + fileid +'_standard_landmarks.txt'

    #%% Landmark align 
    if align == True:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(input_file)
        reader.Update()
        LAA_input = reader.GetOutput()   
                
        landmarks = ((np.loadtxt(LM_file,skiprows=2)))[0:4,:]
        lm_points = vtk.vtkPoints()
        for point in landmarks:
            lm_points.InsertNextPoint(point)
           
        # translate to COM
        com_filter = vtk.vtkCenterOfMass()
        com_filter.SetInputData(LAA_input)
        com_filter.SetUseScalarsAsWeights(False)
        com_filter.Update()     
        com = com_filter.GetCenter()
        
        t1 = vtk.vtkTransform()
        t1.Translate([-c for c in com])
        
        trans_filter = vtk.vtkTransformFilter()
        trans_filter.SetInputData(LAA_input)
        trans_filter.SetTransform(t1)
        trans_filter.Update()
        LAA_transformed = trans_filter.GetOutput()
        
        lm_pd = vtk.vtkPolyData()
        lm_pd.SetPoints(lm_points)
        trans_filter2 = vtk.vtkTransformFilter()
        trans_filter2.SetInputData(lm_pd)
        trans_filter2.SetTransform(t1)
        trans_filter2.Update()
        lm_points2 = trans_filter2.GetOutput().GetPoints()
            
        if fileid == align_to: 
            # Pass without alignment
            source_lm = vtk.vtkPoints()
            source_lm.DeepCopy(lm_points2)
            source_pd = vtk.vtkPolyData()
            source_pd.DeepCopy(LAA_transformed)
        
        else:        
            landmarkTransform = vtk.vtkLandmarkTransform()
            landmarkTransform.SetSourceLandmarks(lm_points2)
            landmarkTransform.SetTargetLandmarks(source_lm)
            #landmarkTransform.SetModeToRigidBody()
            landmarkTransform.SetModeToSimilarity()
            landmarkTransform.Update()
    
            TransformFilter = vtk.vtkTransformPolyDataFilter()
            TransformFilter.SetInputData(LAA_transformed)
            TransformFilter.SetTransform(landmarkTransform)
            TransformFilter.Update()
            LAA_transformed = TransformFilter.GetOutput()
                
            #%% ICP alignment
            icp = vtk.vtkIterativeClosestPointTransform()
            icp.SetSource(LAA_transformed)
            icp.SetTarget(source_pd)
            icp.GetLandmarkTransform().SetModeToSimilarity()
            
            icp.Modified()
            icp.Update()
            #------------------------------------------------------------------------------------------------
            # Apply icp transformation to transformed source surface
            TransformFilter = vtk.vtkTransformPolyDataFilter()
            TransformFilter.SetInputData(LAA_transformed)
            TransformFilter.SetTransform(icp)
            TransformFilter.Update()
            LAA_transformed = TransformFilter.GetOutput()
            
        Save intermediate (for debugging only)
        debug_filename = 'H:/ESOF/align/'+fileid+'.vtk'   
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(debug_filename)
        writer.SetInputData(LAA_transformed)
        writer.Write()
        
        #%% Normalize surface to unit-sphere       
        np_points = vtk_to_numpy(LAA_transformed.GetPoints().GetData())
        scale_factor = [1/np.max(np.sqrt(np.sum(np_points**2,1)))]*3
        sf.append(scale_factor)
        #np.min(np.array(sf)[:,0])
        #scale_factor = (0.005967746885067602,0.005967746885067602,0.005967746885067602)    # Full head
        scale_factor = ( 0.008855780009214411, 0.008855780009214411, 0.008855780009214411)     # Face only
        
        t2 = vtk.vtkTransform()
        t2.Scale(scale_factor)
        
        scale_filter = vtk.vtkTransformFilter()
        scale_filter.SetInputData(LAA_transformed)
        scale_filter.SetTransform(t2)
        scale_filter.Update()
        LAA_scale = scale_filter.GetOutput()
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(norm_file)
        writer.SetInputData(LAA_scale)
        writer.Write()
    else: 
        scale_factor = ( 0.008855780009214411, 0.008855780009214411, 0.008855780009214411)     # Face only
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(norm_file)
        reader.Update()
        LAA_scale = reader.GetOutput()
    
    #%% Find sample points:
    # Sample 250.000 points on the surface
    bb = LAA_scale.GetBounds()[1]-LAA_scale.GetBounds()[0]
    
    #samplingDistance = bb/1000                         # Generates about 1.352.476 points
    samplingDistance = bb/425                           # Generates about 20.000 points
    resampler = vtk.vtkPolyDataPointSampler()
    resampler.GenerateEdgePointsOn()
    resampler.GenerateVertexPointsOn()
    resampler.GenerateInteriorPointsOff()
    resampler.GenerateVerticesOn()
    resampler.SetDistance(samplingDistance)
    resampler.SetInputData(LAA_scale)
    resampler.Update()
    
    resampled_points = vtk_to_numpy(resampler.GetOutput().GetPoints().GetData())
    
    #print("Number of sampled points:")
    #print(resampler.GetOutput().GetNumberOfPoints())
    
    # Pertube points with zero-mean gaussian noise with 
    sigma1 =  (20*scale_factor[0]/2)
    sigma2 = (40*scale_factor[0]/2)
    noise1 = np.random.normal(0.0,sigma1,resampled_points.shape)
    noise2 = np.random.normal(0.0,sigma1,resampled_points.shape)
    
    resampled_points1 = resampled_points+noise1
    resampled_points2 = resampled_points+noise2
    
    # Sample 25.000 points uniformaly within volume 
    nsize = 25000
        
    unit_points = np.zeros((nsize,3)).astype(np.float32)
    # sphere:
    def getPoint():
        d = 100
        while d>1: 
            x = np.random.uniform(-1,1)
            y = np.random.uniform(-1,1)
            z = np.random.uniform(-1,1)
            d = x*x + y*y + z*z
        return x,y,z
    
    for i in range(nsize):
        unit_points[i,0],unit_points[i,1],unit_points[i,2] = getPoint()
    
    all_points = np.vstack([resampled_points1,resampled_points2,unit_points])
    
#%% Find nearest distance to surface 
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(LAA_scale)
    cellLocator.BuildLocator()    
    
    all_sdf = np.zeros(all_points.shape[0]).astype(np.float32)
    for i, point in enumerate(all_points):
        cellId = vtk.reference(0)
        c = [0.0,0.0,0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        cellLocator.FindClosestPoint(point,c,cellId,subId,d)
        all_sdf[i] = np.sqrt(d.get())

    
    final_points = all_points
    final_sdfs = all_sdf
    
    #print("Number of samples in " + fileid + ": ")
    print(final_points.shape[0])
    
    # TODO: Split into two arrays to conform with the SDF code. All distances are positive - we do NOT keep a notion of inside and outside
    npoints = final_points.shape[0]
    pos_points = final_points[0:int(npoints/2),:]
    neg_points = final_points[int(npoints/2)::,:]
    pos_sdf = final_sdfs[0:int(npoints/2)]
    neg_sdf = final_sdfs[int(npoints/2)::]
    
    pos = np.hstack([pos_points,np.expand_dims(pos_sdf,1)])
    neg = np.hstack([neg_points,np.expand_dims(neg_sdf,1)])
    
    #%% Save to npz file where  one can access "pos" and "neg" as sample["pos"] and sample["neg"]
    np.savez(output_file, pos=pos, neg=neg)