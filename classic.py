# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:17:09 2016

@author: M1029148
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import os
os.chdir('D:\\allstate')
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")


ntrain=df_train.shape[0]
ntrainridge=int(round(0.5*ntrain, 0))
ncv=int(round(0.1*ntrain,0))

df_train_ridreg=df_train.iloc[:ntrainridge,:]
df_train_cv=df_train.iloc[ntrainridge:ntrainridge+ncv,:]
df_train_xgboost=df_train.iloc[ntrainridge+ncv:,:]



for c in df_train.columns:
    if 'cat' in c:
        df_train_ridreg[c]=df_train_ridreg[c].astype('category')
        df_train_ridreg[c + '_numeric'] = df_train_ridreg[c].cat.codes
        df_train_cv[c]=df_train_cv[c].astype('category')
        df_train_cv[c + '_numeric'] = df_train_cv[c].cat.codes
        df_train_xgboost[c]=df_train_xgboost[c].astype('category')
        df_train_xgboost[c + '_numeric'] = df_train_xgboost[c].cat.codes
        df_test[c]=df_test[c]. astype('category')   
        df_test[c + '_numeric'] = df_test[c].cat.codes

cont_cols=[]
for c in df_train_ridreg.columns:
    if 'cont' in c:
        cont_cols.append(c)
cat_num_cols=[]
for c in df_train_ridreg.columns:
    if 'numeric' in c:
        cat_num_cols.append(c)

       
        
x_train_ridreg=df_train_ridreg[cat_num_cols]
x_train_cv_rr=df_train_cv[cat_num_cols]
x_train_xgboost=df_train_xgboost[cat_num_cols]
y_train_ridreg=df_train_ridreg['loss']
y_train_cv_rr=df_train_cv['loss']


regr = linear_model.Ridge(alpha = 0.1)
regr.fit(x_train_ridreg,y_train_ridreg)
print('Ridge Accuracy')
print(mean_absolute_error(y_train_cv_rr, regr.predict(x_train_cv_rr)))


x_train_blend=df_train_xgboost[cont_cols]
x_train_blend['ridge_input']=regr.predict(df_train_xgboost[cat_num_cols])
x_test_blend=df_test[cont_cols]
x_test_blend['ridge_input']=regr.predict(df_test[cat_num_cols])



x_train=np.array(x_train_blend)
x_test=np.array(x_train_blend)
y_train=df_train_xgboost['loss']


dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'seed':0,
    'colsample_bytree':0.7,
    'silent':1,
    'subsample':0.7,
    'learning_rate':0.075,
    'objective':'reg:linear',
    'max_depth':6,
    'num_parallel_tree':1,
    'min_child_weight':1,
    'eval_metric':'mae'
}

def xg_eval_mae(yhat,dtrain):
    y=dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y),np.exp(yhat))
res=xgb.cv(xgb_params,dtrain,num_boost_round=750,nfold=4,seed=seed,stratified=False,early_stopping_rounds=15,verbose_eval=10,show_stdv=True,feval=xg_eval_mae,maximize=False)
best_nrounds = res.shape[0] - 1
cv_mean = res.iloc[-1, 0]
cv_std = res.iloc[-1, 1]
print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))


gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
print('Blend Accuracy')
x_train_cv_xgb=df_train_cv[:,cont_cols]
x_train_cv_xgb['ridge_input']=regr.predict(x_train_cv_rr)

print(accuracy_score(y_train_cv_rr, gbdt.predict(x_train_cv_xgb)))
submission = pd.read_csv(submission_file)
submission.iloc[:, 1] = np.exp(gbdt.predict(dtest))
submission.to_csv('xgb_ridge.sub.csv', index=None)












