# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:12:17 2021

@author: kajul
"""
import subprocess
import vtk
import multiprocessing
from tqdm import tqdm
import os
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import math

def mesh_UDF(args):
    input_file = args[0]
    output_file = args[1]
    dir_mrf = args[2]
    
    if os.path.exists(input_file):
        quiet = True
        if quiet:
            output_pipe = open(os.devnull, 'w')       # Ignore text output from MRF.exe.
        else:
            output_pipe = None
            
        subprocess.call([dir_mrf, '-i', input_file, '-o', output_file, '-u', '-I', '0.02'], stdout=output_pipe)

def rotate_vtk(args):
    input_file = args[0]
    output_file = args[1]
    
    if os.path.exists(input_file):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(input_file)
        reader.Update()
        pred = reader.GetOutput()
        
        transFilter = vtk.vtkTransformPolyDataFilter()
        transform = vtk.vtkTransform()
        transform.Scale(1,1,-1)
        transform.RotateX(180)
        transform.RotateY(-90)
        transFilter.SetInputData(pred)
        transFilter.SetTransform(transform)
        transFilter.Update()
        
        pred_flip = transFilter.GetOutput()
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pred_flip)
        writer.SetFileName(output_file)
        writer.Write()
        
def eval_reconstruction(args):
    pred_file = args[0]
    true_file = args[1]
    eval_file = args[2]
    filename = os.path.split(pred_file)[-1][:-4]
    dataset = os.path.split(os.path.split(os.path.split(pred_file)[0])[0])[-1]
    
    if os.path.exists(pred_file):
        reader1 = vtk.vtkPolyDataReader()
        reader1.SetFileName(pred_file)
        reader1.Update()
        pred = reader1.GetOutput()
        
        reader2 = vtk.vtkPolyDataReader()
        reader2.SetFileName(true_file)
        reader2.Update()
        true = reader2.GetOutput()
        
        # Chamfer distance:
        distanceFilter = vtk.vtkDistancePolyDataFilter()
        distanceFilter.SetInputData(0,pred)
        distanceFilter.SetInputData(1,true)
        distanceFilter.ComputeSecondDistanceOn()
        distanceFilter.SignedDistanceOff()
        distanceFilter.Update()
        
        try: 
            vtk_d = distanceFilter.GetOutput().GetPointData().GetScalars()
            chamfer = np.mean(vtk_to_numpy(vtk_d))
        except: 
            print("Something is wrong with the distance filter...")
            return
            
        #MeshAccuracy
        PREDpoints = vtk_to_numpy(pred.GetPoints().GetData())
        
        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(true)
        cellLocator.BuildLocator()
        
        dist = np.zeros(PREDpoints.shape[0])
        for idx in range(PREDpoints.shape[0]):
            testPoint = PREDpoints[idx]
            
            #Find the closest points to TestPoint
            cellId = vtk.reference(0)
            c = [0.0, 0.0, 0.0]
            subId = vtk.reference(0)
            d = vtk.reference(0.0)
            cellLocator.FindClosestPoint(testPoint, c, cellId, subId, d)
            
            dist[idx] = d
            
        meshAcc = math.sqrt(np.percentile(dist,90))

        # Mesh completion: 
        if dataset == "ESOF_NDF": 
            delta = 0.01771156001842882 #2mm
        elif dataset == "EARS_NDF": 
            delta =  0.01857562542972395 #2mm
        elif dataset == "LA_NDF": 
            delta = 0.0151804977301987 #2mm
        else: 
            delta = 0
            
        GTpoints = vtk_to_numpy(true.GetPoints().GetData())
        
        ## Create the tree
        cellLocator = vtk.vtkCellLocator()
        cellLocator.SetDataSet(pred)
        cellLocator.BuildLocator()
        
        dist = np.zeros(GTpoints.shape[0])
        for idx in range(GTpoints.shape[0]):
            testPoint = GTpoints[idx]
            
            #Find the closest points to TestPoint
            cellId = vtk.reference(0)
            c = [0.0, 0.0, 0.0]
            subId = vtk.reference(0)
            d = vtk.reference(0.0)
            cellLocator.FindClosestPoint(testPoint, c, cellId, subId, d)

            # Rasmus 9/4-2020...do remember that d is the squared distance.
            dist[idx] = math.sqrt(d)
        
        MeshCompl = np.sum(dist<delta)/dist.shape[0]
        
        with open(eval_file,'a+') as f: 
            f.write(dataset + "\t")
            f.write(filename+ "\t")
            f.write(str(chamfer) + "\t")
            f.write(str(meshAcc) + "\t")
            f.write(str(MeshCompl) + "\t")
            f.write("\n")
        
        return
        

    

def imap_unordered_bar(func, args, n_processes = 2):
    p = multiprocessing.Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list