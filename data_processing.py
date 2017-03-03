import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file = 'Dataset.csv'

data = pd.read_csv(file)

columns = ['data','Sepal_length','Sepal_width','Petal_length','Petal_width','Class']

data.columns  = columns

data.info()

data = data.drop(['data'],axis = 1)

labels = data['Class']

data = data.drop(['Class'],axis = 1)

dataset = data.values

train_data = dataset[0:100,:]

test_data = dataset[100:,:]

train_labels = labels[0:100]

test_labels = labels[100:]
   
labels_data = labels.values

train_labels = labels_data[0:100]

test_labels = labels_data[100:]

clf = RandomForestClassifier()

clf = clf.fit(train_data,train_labels)

pred = clf.predict(test_data)

print pred

print test_labels

print train_labels