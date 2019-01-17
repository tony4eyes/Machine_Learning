import pandas as pd
df = pd.read_csv('Position_Salaries.csv')

x = df['Level'].values.reshape(df['Level'].size, -1)
y = df['Salary'].values.reshape(df['Salary'].size, -1)

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators = 10)
clf.fit(x,y)

y_pred = clf.predict(x).reshape(df['Salary'].size, -1)

import matplotlib.pyplot as plt
plt.scatter(x, y, color = 'red')
plt.plot(x, y_pred, color = 'blue')
plt.show()

import numpy as np
x_grid = np.arange(x.min(), x.max(), step = 0.1)
x_grid = x_grid.reshape(x_grid.size, -1)
y_grid = clf.predict(x_grid).reshape(len(x_grid),-1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,y_grid,color='blue')
plt.show()
