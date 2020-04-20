#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 05:07:19 2020

@author: vishwa
"""


#data preprocessing
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib qt

#getting data
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

'''from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)'''
#no need of train_test set becz data itself is small and we want accurate predictions


#fitting a linear model
from sklearn.linear_model import LinearRegression
lRegressor = LinearRegression()
lRegressor.fit(X,y)


#fitting apolynomial model
from sklearn.preprocessing import PolynomialFeatures
pRegressor = PolynomialFeatures(degree = 4)
X_poly = pRegressor.fit_transform(X)
pRegressor2 = LinearRegression()
pRegressor2.fit(X_poly,y)


#visualising linear model
plt.scatter(X,y,color = 'red')
plt.plot(X,lRegressor.predict(X),color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualising polynomial model
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y,color = 'red')
plt.plot(X_grid,pRegressor2.predict(pRegressor.fit_transform(X_grid)),color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#predicting a new result with linear regression
lRegressor.predict(6.5)

#predicting a new result with polynomial regression
pRegressor2.predict(pRegressor.fit_transform(6.5))
