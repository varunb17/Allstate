
import os
print(os.getcwd())
os.chdir('/home/varun/allstate')


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
test_x = train_test.iloc[ntrain:,:]

xgdmat=xgb.DMatrix(train_x,train['log_loss'])
params= {'eta':0.1, 'seed':0, 'subsample':0.5, 'colsample_bytree':0.5, 'objectiive': 'reg:linear', 'max_depth':4, 'min_child_weight':3}
num_rounds=100
bst=xgb.train(params, xgdmat, num_boost_round=num_rounds)

test_xgb= xgb.DMatrix(test_x)
submission=pd.read_csv("sample_submission.csv")
submission.iloc[:,1]=np.exp(bst.predict(test_xgb))
submission.to_csv('xgboost.csv', index=None)
