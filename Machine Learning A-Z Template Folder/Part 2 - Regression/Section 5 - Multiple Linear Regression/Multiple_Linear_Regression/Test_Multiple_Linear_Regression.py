#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 05:11:12 2020

@author: vishwa
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
#from pandas.core import datetools

#get data
data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
#from sklearn.compose import ColumnTransformer

#ct = ColumnTransformer(transformers = [("oh",OneHotEncoder(),[3])],remainder='passthrough')

#X  = ct.fit_transform(X)

le = LabelEncoder()
X[:,3]=le.fit_transform(X[:,3])
oh = OneHotEncoder(categorical_features=[3])
X = oh.fit_transform(X).toarray()

#avoiding dummy variable trap
X = X[:,1:]


#splitting training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)


#No need of Feature Scaling as Multiple linear regression library takes care of that



#Fitting the Multiple linear regression  to the traning set
from sklearn.linear_model import LinearRegression
sRegressor = LinearRegression()
sRegressor.fit(x_train,y_train)

#predicting the testset results
y_pred = sRegressor.predict(x_test)


#Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)

X_opt = X[:,[0,1,2,3,4,5]] 
sRegressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
sRegressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]] 
sRegressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
sRegressor_OLS.summary()
    
X_opt = X[:,[0,3,4,5]] 
sRegressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
sRegressor_OLS.summary()

X_opt = X[:,[0,3,5]] 
sRegressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
sRegressor_OLS.summary()

X_opt = X[:,[0,3]] 
sRegressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
sRegressor_OLS.summary()


sRegressor.score(x_test,y_test)
