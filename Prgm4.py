# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:17:09 2016

@author: M1029148
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import os
os.chdir('D:\\allstate')
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
usecols=[]
for c in df_train.columns:
    if 'cont' in c:
        usecols.append(c)

x_train = df_train[usecols]
x_test =df_test[usecols]
y_train=df_train['loss']
id_test=df_test['id']

for c in df_train.columns:
    if 'cat' in c:
        df_train[c]=df_train[c].astype('category')
        df_test[c]=df_test[c]. astype('category')
        x_train[c + '_numeric'] = df_train[c].cat.codes
        x_test[c + '_numeric'] = df_test[c].cat.codes

regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pred = regr.predict(x_test)
sub= pd.DataFrame()
sub['id']=id_test
sub['loss']=y_pred
sub.to_csv('lin_regression.csv', index=False)