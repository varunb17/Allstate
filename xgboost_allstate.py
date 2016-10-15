import os
os.chdir('/home/varun/allstate')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

id='id'
target='loss'
seed=0
data_dir='/home/varun/allstate'

train_file = "{0}/train.csv".format(data_dir)
test_file="{0}/test.csv".format(data_dir)
submission_file="{0}/sample_submission.csv".format(data_dir)

train=pd.read_csv(train_file)
test=pd.read_csv(test_file)

y_train=np.log(train[target].ravel())

train.drop([id,target],axis=1,inplace=True)
test.drop([id],axis=1,inplace=True)

ntrain=train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)

features=train.columns

cats = [feat for feat in features if 'cat' in feat]
for feat in cats:
    train_test[feat]=pd.factorize(train_test[feat],sort=True)[0]



x_train=np.array(train_test.iloc[:ntrain,:])
x_test=np.array(train_test.iloc[ntrain:,:])


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
submission = pd.read_csv(submission_file)
submission.iloc[:, 1] = np.exp(gbdt.predict(dtest))
submission.to_csv('xgb_1.sub.csv', index=None)

