#data preprocessing


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
data = pd.read_csv('Data.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,3].values


#taking care of misssing data
'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])'''


#Encoding the categorical data
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:, 0])
le.classes_
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
y = le.fit_transform(y)'''


#Splittting data into training and test set
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test =  train_test_split(X, y, test_size = 0.2, random_state = 0)



#feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''



