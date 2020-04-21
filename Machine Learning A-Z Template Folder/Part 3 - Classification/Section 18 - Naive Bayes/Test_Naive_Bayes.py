#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:18:34 2020

@author: vishwa
"""


#Naive_Bayes_Algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:,2:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state =0);

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from matplotlib.colors import ListedColormap
x_set,y_set = x_train,y_train
X1,X2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1, stop = x_set[:,0].max()+1,step = 0.01),
                    np.arange(start = x_set[:,1].min()-1, stop = x_set[:,1].max()+1,step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c = ListedColormap(('red','green'))(i),label = j)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Naive Bayes Classification(Training set)')
#plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
X1,X2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1, stop = x_set[:,0].max()+1,step = 0.01),
                    np.arange(start = x_set[:,1].min()-1, stop = x_set[:,1].max()+1,step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c = ListedColormap(('red','green'))(i),label = j)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Naive Bayes Classification(Test set)')
#plt.legend()
plt.show()


