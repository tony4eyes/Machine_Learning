import pandas as pd

df = pd.read_csv('Social_Network_Ads.csv')

x = df[['Age','EstimatedSalary']].values

y = df['Purchased'].values.reshape(len(x), -1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
x_train = sd.fit_transform(x_train)
x_test = sd.transform(x_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 0)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test).reshape(len(y_test),-1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


import matplotlib.pyplot as plt
