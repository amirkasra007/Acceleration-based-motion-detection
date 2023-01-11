# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:39:19 2022

@author: csd81363
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pickle
import json
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, mutual_info_regression
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pyts.classification import TSBF

from scipy.spatial.distance import euclidean


#%% Importing Dataset
results = r"C:\Motion Based Acceleration Detection\Excel_Files\Results"
df_main = pd.read_excel(r"C:\Motion Based Acceleration Detection\Excel_Files\Subjects_14To26\data_1614_accel_phone.xlsx")
x_data = df_main[["X","Y","Z"]]
y_data = df_main["Act"]
data_columns = ['X', 'Y', 'Z']

#%% Rolling Window
df_main.set_index("Timestamp", inplace= True)
start = df_main.index.values[0]
end = df_main.index.values[-1]
rolloff = df_main[data_columns].rolling(7, center=True).mean()
rolloff_median = df_main[data_columns].rolling(7, center=True).median()
rolloff_std = df_main[data_columns].rolling(7, center=True).std()
rolloff.head(10)
fig, ax = plt.subplots()
fig, ax2 = plt.subplots()
fig, ax3 = plt.subplots()
'''
for var in data_columns:
    ax.plot(rolloff.loc[start:end, 'Z'], marker='.',color= "green", markersize=2,label='7-marker')
    ax.set_title('Rolling Mean of Z')
    ax2.plot(rolloff_median.loc[start:end, 'Z'], marker='.',color= "blue", markersize=2, label='7-marker')
    ax2.set_title('Rolling Median of Z')
    ax3.plot(rolloff_std.loc[start:end, 'Z'], marker='.',color= "red", markersize=2,  label='7-marker')
    ax3.set_title('Rolling Standard Deviation of Z')
'''
#%% Dickey Fuller Test for Stationarity

adft = adfuller(df_main['Z'].loc[df_main['Act'] == 'R'],autolag="AIC")
output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used", 
                                                        "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
#print(output_df)

# Not stationary for X as our p-value is greater than 5 percent and the test statistic is greater than the critical value.
# Stationary for Y as our p-value is lesser than 5 percent and the test statistic is lesser than the critical value.
# Not stationary for Z as our p-value is greater than 5 percent and the test statistic is greater than the critical value.

#%% AutoCorrelation
'''
autocorrelation_lag7 = df_main['Z'].loc[df_main['Act'] == 'A'].autocorr(lag=7)
print("7 period Lag for Act = A and acc = Z: ", autocorrelation_lag7)
autocorrelation_lag20 = df_main['Z'].loc[df_main['Act'] == 'B'].autocorr(lag=20)
print("20 period Lag for Act = B and acc = Z ", autocorrelation_lag20)
autocorrelation_lag50 = df_main['Z'].loc[df_main['Act'] == 'P'].autocorr(lag=50)
print("50 period Lag for Act = P and acc = Z: ", autocorrelation_lag50)
autocorrelation_lag100 = df_main['Z'].loc[df_main['Act'] == 'R'].autocorr(lag=100)
print("100 period Lag for Act = R and acc = Z: ", autocorrelation_lag100)
'''
#%% Seasonal Decompose

decompose = seasonal_decompose(df_main['Z'].loc[df_main['Act'] == 'R'],model='additive', period=20)
decompose.plot()
plt.show()
#%% Feature Selection
k=3
X_best = SelectKBest(score_func= f_classif, k=k).fit_transform(x_data,y_data)
selector = SelectKBest(f_classif, k=k)
selector.fit(x_data, y_data)
names = selector.feature_names_in_.tolist()
#print(names)
mask = selector.get_support()
new_features = x_data.columns[mask]
#%% Time Series Bag-Of-Features
random_state = 0
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X_best, y_data, test_size=test_size, random_state=random_state)

clf = TSBF(bins=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
something= f1_score(y_test, y_pred, average='micro')
print(something)

'''
plt.bar(np.arange(clf.n_features_in_), clf.feature_importances_)
plt.title('Feature importance scores')
plt.xticks(np.arange(clf.n_features_in_),
           ['feature {}'.format(i) for i in range(clf.n_features_in_)],
           rotation=90)
plt.ylabel("Mean decrease in impurity")
plt.tight_layout()
plt.show()
'''
#%% Random Forest Classifier
'''
random_state = 0
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X_best, y_data, test_size=test_size, random_state=random_state)

classif_model = RandomForestClassifier();
classif_model.fit(X_train, y_train)

y_pred = classif_model.predict(X_test)
something= f1_score(y_test, y_pred, average='micro')
print(something)
'''
#%% Plots
'''
plt.figure(figsize=(10,6))
plt.plot(y_pred,  color="orange")
plt.plot(y_test, color="blue")
plt.ylabel('Target value')
plt.xlabel('Samples (ordered ascendingly by y_test)')
plt.title('y_pred vs. y_test: ')
plt.show()
'''
#%% Save Model

save_stuff = results + "\model" + ".sav"
pickle.dump(clf, open(save_stuff, 'wb'))

#%% Testing Model
to_test = r"C:\Motion Based Acceleration Detection\Excel_Files\Results\ForTestingPurposes.xlsx"
df_test = pd.read_excel(to_test)
'''
values =df_test.iloc[:,:].tolist()
num= np.array(values).reshape(1,-1)
print(num)
'''
model_path = r"C:\Motion Based Acceleration Detection\Excel_Files\Results\model.sav"
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)
    what_is_the_result= loaded_model.predict(df_test)
    print(what_is_the_result)

# https://builtin.com/data-science/time-series-python
# https://hal.inria.fr/hal-03558165/document ----> "Time series bag-of-features"