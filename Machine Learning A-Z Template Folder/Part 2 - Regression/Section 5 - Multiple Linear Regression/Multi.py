# -*- coding: utf-8 -*-
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

import sklearn.preprocessing as pre

lab = pre.LabelEncoder()
hot = pre.OneHotEncoder( categorical_features = [3] )
X[:, 3] = lab.fit_transform(X[:, 3])
X = hot.fit_transform(X).toarray()

import sklearn.cross_validation as cv
X_train, Y_train, X_test, Y_test = cv.train_test_split(X, Y, test_size = 0.2, random_seed = 0)

