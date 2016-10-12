# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:03:32 2016

@author: M1029148
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir('D:\\allstate')


data_train_raw = pd.read_csv('train.csv')
data_test_raw =pd.read_csv("test.csv")


col_uniques=[]
for col in data_train_raw.columns:
    if (col.find('cat')!=-1):
        col_uniques.append([col, len(data_train_raw[col].unique())])
print(col_uniques)

      

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in data_train_raw.columns:
    if (col.find('cat') !=-1):
        print(col)
        data_train_raw[str(col+'_numerical')]=le.fit_transform(data_train_raw[col])
        data_test_raw[col] = data_test_raw[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
        le.classes_ = np.append(le.classes_, '<unknown>')
        data_test_raw[str(col+'_numerical')]=le.transform(data_test_raw[col])
print(data_train_raw.columns)

