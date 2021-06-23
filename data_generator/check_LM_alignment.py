# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:25:21 2021

@author: kajul
"""

import vtk 
import numpy as np
import os


files = os.listdir('H:/ESOF/Cut/')

fileids = []
for file in files:
    fileids.append(file[0:14])

    
    
#%%
for fileid in fileids: 
    print(fileid)
    surfacepath = 'H:/ESOF/Cut/'+fileid+'_standard_cut.vtk'
    lmpath = 'M:/ESOF/'+fileid+ '/'+fileid+'_standard_landmarks.txt'
    
    #LM
    npLandmarks = np.loadtxt(lmpath,skiprows = 0)   
    vtkLandmarks = vtk.vtkPoints()
    for landmark in npLandmarks:
        vtkLandmarks.InsertNextPoint(landmark)
    vtkPolyDataLandmarks = vtk.vtkPolyData()
    vtkPolyDataLandmarks.SetPoints(vtkLandmarks)
    
    vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
    vertexGlyphFilter.AddInputData(vtkPolyDataLandmarks)
    vertexGlyphFilter.Update()
    mapper1 = vtk.vtkPolyDataMapper()
    mapper1.SetInputConnection(vertexGlyphFilter.GetOutputPort())
    actor1 = vtk.vtkActor()
    actor1.SetMapper(mapper1)
    actor1.GetProperty().SetPointSize(10)
    actor1.GetProperty().SetColor(1,0,0)
    
    # surface
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(surfacepath)
    reader.Update()
    surface = reader.GetOutput()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(800,600)
    renderWindow.SetWindowName("VTK")
    
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(actor1)  
    renderer.SetBackground(1,1,1)
    
    renderWindow.AddRenderer(renderer)
    renderWindow.Render()
    renderWindowInteractor.Start()    
    
#%% Check scaled surfaces
list_of_fileids = 'H:/ESOF/ESOF_ids.txt'
basepath = 'E:/DATA/ESOF/mesh_normalized/'
align_to = '20140621103743'
fileids = []
f =  open(list_of_fileids,"r")
for x in f:
    if not x.strip == align_to:
        fileids.append(x.strip())    

align_path = basepath + align_to + ".vtk"
reader = vtk.vtkPolyDataReader()
reader.SetFileName(align_path)
reader.Update()
align = reader.GetOutput() 
mapper1 = vtk.vtkPolyDataMapper()
mapper1.SetInputData(align)
actor1 = vtk.vtkActor()
actor1.SetMapper(mapper1)

for fileid in fileids: 
    surfacepath = basepath + fileid + ".vtk"
    
    textActor = vtk.vtkTextActor()
    textActor.SetInput ( fileid )
    textActor.SetPosition2 ( 10, 40 )
    textActor.GetTextProperty().SetFontSize ( 24 )
    textActor.GetTextProperty().SetColor ( 1.0, 0.0, 0.0 )
    
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(surfacepath)
    reader.Update()
    surface = reader.GetOutput()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,0,0)
    
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(800,600)
    renderWindow.SetWindowName("VTK")
    
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor(actor1)  
    renderer.AddActor2D ( textActor )
    renderer.SetBackground(1,1,1)
    
    renderWindow.AddRenderer(renderer)
    renderWindow.Render()
    renderWindowInteractor.Start()    
    