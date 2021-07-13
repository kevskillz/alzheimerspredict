import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler


import pickle
import numpy as np

path = 'oasis_longitudinal.csv'

df = pd.read_csv(path)
df = df[['M/F', 'Age', 'EDUC', 'CDR', 'ASF', 'eTIV']]

refDict = {0: "No Alzheimer's", 2: "Mild Alzheimer's", 1: "Very Mild Alzheimer's", 3: "Moderate Alzheimer's"}



df['M/F'] = [1 if i == 'M' else 0 for i in df['M/F']] # male == 1, female == 0

df.CDR = [3 if i == 2 else i for i in df.CDR]
df.CDR = [2 if i == 1 else i for i in df.CDR]
df.CDR = [1 if i == 0.5 else i for i in df.CDR]

ROS = RandomOverSampler() # uneven data, helps balance it



X = df.drop(['CDR'], axis=1, inplace=False)
X['M/F'] = X['M/F'].astype('int8')
X.Age = X.Age.astype('int8')
X.EDUC = X.EDUC.astype('int8')
X.eTIV = X.eTIV.astype('int16')
X.ASF = X.ASF.astype('float16')

y = df.CDR.astype('int8')

X, y = ROS.fit_resample(X, y)
#print(y)
#print(X)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)



model = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
model.fit(X_train, y_train)
print(X_test)
print(y_test)
prediction = model.predict(X_test)

print(model.score(X_test, y_test))
print(confusion_matrix(y_test, prediction))

with open(f'alzheimersDemographic.model', 'wb') as file:
    pickle.dump(model, file)


