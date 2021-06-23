# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:08:58 2021

@author: kajul
"""

import vtk
import numpy as np
import os


surface_name = 'E:/DATA/ESOF/mesh_normalized/20140621103743.vtk'
samplings_name = 'E:/DATA/ESOF/NDF_samples_old/20140621103743.npz'

samples = np.load(samplings_name)

s_show1 = samples["pos"][0::100,0:3]
s_show2 = samples["neg"][0::100,0:3]
s_show = np.vstack((s_show1,s_show2))
print(s_show.shape)

reader = vtk.vtkPolyDataReader()
reader.SetFileName(surface_name)
reader.Update()
surf = reader.GetOutput()

s_points = vtk.vtkPoints()
for point in s_show:
    s_points.InsertNextPoint(point)
vtkPolyDataLandmarks = vtk.vtkPolyData()
vtkPolyDataLandmarks.SetPoints(s_points)
    
vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
vertexGlyphFilter.AddInputData(vtkPolyDataLandmarks)
vertexGlyphFilter.Update()
mapper1 = vtk.vtkPolyDataMapper()
mapper1.SetInputConnection(vertexGlyphFilter.GetOutputPort())
actor1 = vtk.vtkActor()
actor1.SetMapper(mapper1)
actor1.GetProperty().SetPointSize(5)
actor1.GetProperty().SetColor(1,0,0)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(surf)
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