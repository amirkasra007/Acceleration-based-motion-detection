"""
Created on Wed Jan  1 15:49:37 2023

@author: Amirkasra007
"""



import pandas as pd
import numpy as np

###importing dataset
df = pd.read_csv('Data/data_1601_accel_phone.csv')
# df['time_index'] =np.arange(0,len(df)/25, 0.04)

# df.set_index('time_index')
X = df.iloc[:, 3:].values
y = df.iloc[:, 1].values


###Taking Care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)


###Encoding the dependant variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

###Splitting dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size = 0.2, random_state = 1)

##Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
    