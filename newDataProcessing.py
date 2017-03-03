import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd 

file = 'Dataset.csv'

df = pd.read_csv(file)

'''df.columns = ['Data','Sepal_length','Sepal_width','Petal_length','Petal_width','Class']

df  = df.drop(['Data'],axis = 1)'''

'''print df.head(10)'''

Y = []

Y = df['Iristype']

df = df.drop(['Iristype'],axis=1)

X = df.values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.8)

clf = RandomForestClassifier()

clf = clf.fit(X_train,Y_train)

pred = clf.predict(X_test)

print str(accuracy_score(pred,Y_test)*100) + "%"