import pandas as pd
df = pd.read_csv('Position_Salaries.csv')

x = df['Level'].values.reshape(df.shape[0],-1)
y = df['Salary'].values.reshape(df.shape[0],-1)

from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()
clf.fit(x,y)

y_pred = clf.predict(x)

import matplotlib.pyplot as plt
plt.scatter(x,y, color = 'red')
plt.plot(x,y_pred, color = 'blue') 
plt.show()
import numpy as np
x_grid = np.arange(x.min(), x.max(), step = 0.1)
x_grid = x_grid.reshape(x_grid.shape[0],-1)
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, clf.predict(x_grid), color = 'blue') 
plt.show()