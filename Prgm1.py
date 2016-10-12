# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:13:15 2016

@author: M1029148
"""

import os
print(os.getcwd())
os.chdir('D:\\allstate')


import numpy as np
import pandas as pd
import xgboost as xgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


features = [x for x in train.columns if x not in ['id','loss']]
cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id', 'loss']]

from scipy.stats import norm, lognorm
import matplotlib.mlab as mlab
train['log_loss']=np.log(train['loss'])

ntrain=train.shape[0]
ntest=test.shape[0]
train_test=pd.concat((train[features],test[features])).reset_index(drop=True)
for c in range(len(cat_features)):
    train_test[cat_features[c]]=train_test[cat_features[c]].astype('category').cat.codes
    
train_x= train_test.iloc[:ntrain,:]

xgdmat=xgb.DMatrix(train_x,train['log__loss'])
params= {'eta':0.01, 'seed':0, 'subsample':0.5, 'colsample_bytree':0.5, 'objectiive': 'reg:linear', 'max_depth':6, 'min_child_weight':3}
num_rounds=1000
bst=xgb.train(params, xgdmat, num_boost_round=num_rounds)

import operator

def create_feature_map(features):
    outfile = open('xgb.fmap','w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
        i=i+1
    outfile.close()
    
create_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df=pd.DataFrame(importance, columns=['feature','fscore'])
df['fscore']=df['fscore']/df['fscore'].sum()
plt.figure
df.plot()
df.plot(kind='barh',x='feature',y='fscore',legent=False,figsize=(6,10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
df
test_xgb= xgb.DMatrix(test_x)
submission=pd.read_csv("sample_submission.csv")
submission.iloc[:,1]=npexp(bst.predict(test_xgb))
submission.to_csv('xgb_starter.sub.csv', index=None)




        
