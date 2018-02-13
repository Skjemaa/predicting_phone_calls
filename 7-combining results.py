# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 08:23:53 2016

@author: user
"""
import pandas as pd
import numpy as np
# we load the random forest and the xgboost results
result1 = pd.read_csv('random_forest.txt',dtype={'DATE':'category','ASS_ASSIGNMENT':'category','prediction': np.int16,} ,sep='\t',parse_dates=[0])
result2 = pd.read_csv('gg_xgboost_300est.txt',dtype={'DATE':'category','ASS_ASSIGNMENT':'category','prediction': np.int16,} ,sep='\t',parse_dates=[0])

result1=result1['prediction']
result2=result2['prediction']

# we see clearly 2 different part on the plots:
#In the left part of plot, we see that the prediction of the random forest is usually too much higher than the prediction with xgboost
# we also see a clear variation of the oscilation amplitude using the xgboost, but not with the random forest
# We decided to see the effect of combining the results of the 2 predictors, based on these observation

# we keep the  random forest  results, except when the random forest results are much higher  ( above a threshold) than the xgbost result,
#In that case, we use the prediction provided by xgboost

index=np.where(result1-result2 > 120 )
result1.iloc[index]=result2.iloc[index]

submission = pd.read_csv('submission.txt' ,sep='\t',parse_dates=[0])

submission.prediction=result1*1.1758
submission.to_csv('final_submission.txt',sep='\t',date_format='%Y-%m-%d %H:%M:%S.000',index=False ,encoding='utf-8')
