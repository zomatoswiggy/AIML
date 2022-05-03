# finding current directory

import os

cwd=os.getcwd()

print(cwd)

# Import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # optional
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

#Importing Data, Viewing the Structure (column name/attribute), Finding size of data

df= pd.read_csv('/content/diabetes.csv')
df.head()
df.shape
# Statistics of Data
df.describe()
#Finding the number of members in the class
df('Outcome').value_counts()
df.mean()
## Independent And dependent Column
X = df.drop(columns='Outcome',axis=1).values
## dropping the label called diabetes in column; axis=0 row, axis=1 column
Y= df['Outcome']
#Standardization of Dataset

Sc=StandardScaler()
Sc.fit(X)
# Transferring all data in common(same) range
Sdat= Sc.transform(X)
Sdat= Sc.transform(X)
print(Sdat)
#new data
X=Sdat
Y=df['Outcome']
#Train Test Split 30% of data(0.3) stratify maintain the ratio of y in input,
#nonreplication of code =1

X_train, X_test, Y_train, Y_test =model_selection.train_test_split(X, Y,
test_size=0.3,stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape )

#Holdout Scheme

model = LogisticRegression()

model.fit(X_train, Y_train)

TP=model.predict(X_train)

print(TP)

TsTP=model.predict(X_test)

print(TsTP)

result = model.score(X_test, Y_test)

print("Accuracy: %.2f%%" % (result*100.0))

# K Fold Cross validation

kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

m_kfold = LogisticRegression()

r_kfold = model_selection.cross_val_score(m_kfold, X, Y, cv=kfold)

print(r_kfold)

print("Accuracy: %.2f%%" % (r_kfold.mean()*100.0))