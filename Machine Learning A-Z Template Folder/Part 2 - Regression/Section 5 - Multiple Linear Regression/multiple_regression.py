import numpy as np
import pandas as pd
from sklearn import preprocessing

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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
clf = LinearRegression()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

import statsmodels.formula.api as sm

#adding constant - y = b0 + x1*b1 + x2*b2...
x = np.append(np.ones((50, 1), dtype = int), x, axis = 1)  # axis - dimension, 0 - row, 1 - column

x_opt = x
#optimize by backward elimination
clf_OLS = sm.OLS(endog = y, exog = x_opt).fit()
clf_OLS.summary()  # drop the highest p value's first

x_opt = x[:, [0,1,3,4,5]]
clf_OLS = sm.OLS(endog = y, exog = x_opt).fit()
clf_OLS.summary()

x_opt = x[:, [0,3,4,5]]
clf_OLS = sm.OLS(endog = y, exog = x_opt).fit()
clf_OLS.summary()

x_opt = x[:, [0,3,5]]
clf_OLS = sm.OLS(endog = y, exog = x_opt).fit()
clf_OLS.summary()

x_opt = x[:, [0,3]]
clf_OLS = sm.OLS(endog = y, exog = x_opt).fit()
clf_OLS.summary()

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)
clf_opt = LinearRegression()
clf_opt.fit(x_train, y_train)
y_pred_opt = clf_opt.predict(x_test)