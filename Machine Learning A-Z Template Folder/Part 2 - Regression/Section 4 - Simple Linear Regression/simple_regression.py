import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('Salary_Data.csv')
"""x = np.array(df['YearsExperience'])
x = x.reshape(-1,1)
y = np.array(df['Salary'])
y = y.reshape(-1,1)"""
x = df.iloc[:,:-1]
y = df.iloc[:,1:1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling, regression takes care of the scaling
#x = preprocessing.scale(x)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, clf.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, clf.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_pred, color = 'red')
plt.scatter(x_test, y_test, color = 'blue')
plt.show()