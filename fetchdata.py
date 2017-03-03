from pandas import read_csv
import pandas as pd
import csv

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
db=pd.read_csv(url, names=['Sepal length','Sepal width','Petal length','Petal width','Iristype'])
db.to_csv('Dataset.csv')


