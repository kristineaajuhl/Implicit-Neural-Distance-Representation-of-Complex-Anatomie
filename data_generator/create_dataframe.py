# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:10:25 2021

@author: kajul
"""

import pandas as pd

path = 'C:/Users/kajul/Documents/Data/ESOF_statistics.xlsx'

esof_pd = pd.read_excel(path,header=0,index_col="ID",usecols=(1,2,3),nrows=435)
esof_pd = esof_pd.rename(columns = {"Gender (1 = male)": "Gender"})
esof_pd["Gender"] = esof_pd["Gender"].fillna(0)
esof_pd.index = esof_pd.index.map(str)

esof_pd.to_pickle('H:/ESOF/ESOF_dataframe.pkl')

df = pd.read_pickle('H:/ESOF/ESOF_dataframe.pkl')
