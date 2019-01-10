import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('50_Startups.csv')

stat = df.describe()

#Encoding features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encode = LabelEncoder()

df['State'] = encode.fit_transform(df['State'])

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

oneHot = OneHotEncoder(categorical_features = [3], sparse = False)
x = oneHot.fit_transform(x)

# Avoiding the Dummy Variable Trap
x = x[:, 1:]

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x1 = sc_x.fit_transform(x)"""
