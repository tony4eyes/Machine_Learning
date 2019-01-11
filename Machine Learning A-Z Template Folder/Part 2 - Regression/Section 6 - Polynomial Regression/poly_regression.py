import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')

#x = df['Level'].values
x = df['Level'].values
x = x.reshape(-1,1)
y = df['Salary'].values
y = y.reshape(-1,1)


from sklearn.linear_model import LinearRegression
#simple LinearRegression model
clf = LinearRegression()
clf.fit(x, y)
y_pred = clf.predict(6.5)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
x_poly = poly.fit_transform(x)

clf_poly = LinearRegression()
clf_poly.fit(x_poly, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, clf.predict(x), color = 'blue')
plt.show()

plt.scatter(x, y, color = 'red')
plt.plot(x, clf_poly.predict(x_poly), color = 'blue')
plt.show()

x_grid = np.arange(x.min(), x.max(), step = 0.1)
x_grid = x_grid.reshape(x_grid.size, 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, clf_poly.predict(poly.transform(x_grid)), color = 'blue')
plt.show()

y_pred = clf.predict(6.5)
y_poly_pred = clf_poly.predict(poly.transform(6.5))