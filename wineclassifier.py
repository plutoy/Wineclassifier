import pandas as pd
from numpy import *
import os

df=os.path.join("wine.csv")
raw_data=pd.read_csv(df)

print(raw_data[0:10])

print(raw_data['alcohol'][0:10])
X=raw_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']].values
raw_data.columns
y=raw_data[['quality']].values


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
transformer=SelectKBest(score_func=chi2,k=5)

Xt_chi2=transformer.fit_transform(X,y)
print(Xt_chi2[0:5])
print(transformer.scores_)

x=[]
for i in range(len(X)):
    x.append([X[i][0],X[i][1],X[i][2],X[i][6],X[i][10]])
    
print(x[0:5])

lb=[]
for i in range(len(y)):
    lb.append(y[i][0])
    
print(lb[0:5])

from sklearn.preprocessing import MinMaxScaler
x=MinMaxScaler().fit_transform(x)

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

dt=DecisionTreeClassifier()
svr=SVC()
knn=KNeighborsClassifier()

print(Xt_chi2.shape)
score=cross_val_score(dt,x,lb,cv=10,scoring='accuracy')

print(score)

score2=cross_val_score(svr,x,lb,cv=10,scoring='accuracy')
print(score2)
