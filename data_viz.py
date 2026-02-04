import numpy as np
import pandas as pd # for data analysis
import seaborn as sns # for vizualisation
import matplotlib.pyplot as plt # for vizualisation
import matplotlib.pylab as pylab

data = pd.read_csv('CW1_train.csv')

categorical_features = ['cut', 'color', 'clarity']
numerical_features = ['carat', 'depth', 'price', 'table']
xyz_features = ['x', 'y', 'z']
a_features = ['a1', 'a2', 'a3', 'a4', 'a5']
b_features = ['b1', 'b2', 'b3', 'b4', 'b5']
a2_features = ['a6', 'a7', 'a8', 'a9', 'a10']
b2_features = ['b6', 'b7', 'b8', 'b9', 'b10']

data[numerical_features + xyz_features + ['outcome']].describe()
data[a_features + b_features].describe()
data[a2_features + b2_features].describe()

data = data.drop(data[data['x']==0].index)
data = data.drop(data[data['y']==0].index)
data = data.drop(data[data['z']==0].index)
data.shape

#idrk