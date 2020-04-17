#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:38:52 2020

@author: vishwa
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#reading data
data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values


#splitting the data into train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)
 
 
#feature scaling is done by the simple linear regression library..so we skip it and start 
#fitting Simple Linear Regression to our training set

from sklearn.linear_model import LinearRegression
sRegressor = LinearRegression() 
sRegressor.fit(x_train,y_train)


#predict the TestSet results
y_pred = sRegressor.predict(x_test)


#Visualising the Training set results
plt.subplot(1,2,1)
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,sRegressor.predict(x_train),color = 'blue')
plt.title("Years of experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
    
#visualising the Test Set Results
plt.subplot(1,2,2)
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,sRegressor.predict(x_train),color = 'blue')
plt.title("Years of experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")


sRegressor.score(x_test,y_test)