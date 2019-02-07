import pandas as pd
df = pd.read_csv('Social_Network_Ads.csv')

x = df.iloc[:,2:4]
y = df['Purchased'].values.reshape(len(df),-1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
st_x_train = st.fit_transform(x_train)
st_x_test = st.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
clf.fit(st_x_train, y_train)

y_pred = clf.predict(st_x_test)
