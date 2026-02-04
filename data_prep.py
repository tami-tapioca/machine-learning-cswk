import numpy as np
import pandas as pd # for data analysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

data = pd.read_csv('CW1_train.csv')
#data.head()

#data.shape
#data.info()

categorical_features = ['cut', 'color', 'clarity']
numerical_features = ['carat', 'depth', 'price', 'table']
xyz_features = ['x', 'y', 'z']
a_features = ['a1', 'a2', 'a3', 'a4', 'a5']
b_features = ['b1', 'b2', 'b3', 'b4', 'b5']
a2_features = ['a6', 'a7', 'a8', 'a9', 'a10']
b2_features = ['b6', 'b7', 'b8', 'b9', 'b10']

# dropping values that equal 0 in x, y and z
data = data.drop(data[data['x']==0].index)
data = data.drop(data[data['y']==0].index)
data = data.drop(data[data['z']==0].index)
#data.shape

"""
shade = ["#835656", "#baa0a0", "#ffc7c8", "#a9a798", "#65634a"]
ax=sns.pairplot(data, hue= "cut", palette=shade)
"""
# dropping outliers
data = data[(data["depth"]>52.5)]
data = data[(data["carat"]<3.5)]
data = data[(data["table"]<70)&(data["table"]>45)]
data = data[(data["y"]<50)]
data = data[(data["z"]<7)&(data["z"]>2)]
#data.shape

label_data = data.copy()

# apply label encoder to categorical data, this will be converted to numbers
# try out one hot encoding also?
label_encoder = LabelEncoder()
for col in categorical_features:
    label_data[col] = label_encoder.fit_transform(label_data[col])
label_data.head()

X, y = label_data[['depth', 'table', 'a1', 'a4', 'b1', 'b3']].values, label_data['outcome'].values

# return x and y