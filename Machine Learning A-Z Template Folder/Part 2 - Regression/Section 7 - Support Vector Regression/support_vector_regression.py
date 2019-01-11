import pandas as pd
df = pd.read_csv('Position_Salaries.csv')

x = df['Level'].values
x = x.reshape(10,-1)

y = df['Salary'].values
y = y.reshape(10,-1)

#SVR has no feature scaling, as SVR is less common model
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x1 = sc_x.fit_transform(x)
y1 = sc_y.fit_transform(y)

from sklearn.svm import SVR
clf = SVR(kernel = 'rbf')
clf.fit(x1, y1)

y_pred = clf.predict(x1)
y_pred = sc_y.inverse_transform(y_pred)


import matplotlib.pyplot as plt

plt.scatter(x, y, color = 'red')
plt.plot(x, y_pred, color = 'blue')

import numpy as np
x_grid = np.arange(x1.min(), x1.max(), step = 0.01)
x_grid = x_grid.reshape(x_grid.size, -1)
plt.scatter(x1, y1, color = 'red')
plt.plot(x_grid, clf.predict(x_grid), color = 'blue')