import numpy as np
import pandas as pd
import datetime

import time as t
# not all the following imports are used in this file
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from math import log,sqrt,exp

# using timer to evaluate the time of execution
start = t.time()
# the functions  to extract feature from Date column . Already used on train data( except for "vacances" and "saison"), but we still need to use them on test data
def year(date) :
    return date.year
def month(date) :
    return date.month
    
def day(date) : 
    return date.day
    
def weekday(date) : 
    return date.dayofweek
    
def weekend(date) : 
    if(weekday(date) >= 5) :
        return 1
    else :
        return 0

    
def time(date) :
    return(date.hour+date.minute/60.0)

def night(date) : 
    t = time(date)
    if (t >= 23.5 or t <= 7):
        return 1
    else :
        return 0
def jours_ferie(date) :
    if( month(date)==1 and day(date)==1):
        return 1
    if(month(date)==5 and day(date)==1):
        return 1
    if(month(date)==5 and day(date)==8):
        return 1   
    if(month(date)==7 and day(date)==14):
        return 1      
    if(month(date)==8 and day(date)==15):
         return 1      
    if(month(date)==11 and day(date)==1):
        return 1
    if(month(date)==11 and day(date)==11):
        return 1
    if(month(date)==12 and day(date)==25):
        return 1
    if(year(date)==2011):
        if(month(date)==4 and day(date)==25):
            return 1
        if(month(date)==6 and day(date)==2):
            return 1
        if(month(date)==6 and day(date)==12):
            return 1
    if(year(date)==2012):
        if(month(date)==4 and day(date)==9):
            return 1
        if(month(date)==5 and day(date)==17):
            return 1
        if(month(date)==5 and day(date)==27):
            return 1
    if(year(date)==2013):
        if(month(date)==4 and day(date)==1):
            return 1
        if(month(date)==5 and day(date)==9):
            return 1
        if(month(date)==5 and day(date)==19):
            return 1
    return 0
def jours_ferie(date) :
    if( month(date)==1 and day(date)==1):
        return 1
    if(month(date)==5 and day(date)==1):
        return 1
    if(month(date)==5 and day(date)==8):
        return 1   
    if(month(date)==7 and day(date)==14):
        return 1      
    if(month(date)==8 and day(date)==15):
         return 1      
    if(month(date)==11 and day(date)==1):
        return 1
    if(month(date)==11 and day(date)==11):
        return 1
    if(month(date)==12 and day(date)==25):
        return 1
    if(year(date)==2011):
        if(month(date)==4 and day(date)==25):
            return 1
        if(month(date)==6 and day(date)==2):
            return 1
        if(month(date)==6 and day(date)==12):
            return 1
    if(year(date)==2012):
        if(month(date)==4 and day(date)==9):
            return 1
        if(month(date)==5 and day(date)==17):
            return 1
        if(month(date)==5 and day(date)==27):
            return 1
    if(year(date)==2013):
        if(month(date)==4 and day(date)==1):
            return 1
        if(month(date)==5 and day(date)==9):
            return 1
        if(month(date)==5 and day(date)==19):
            return 1
    return 0
    
    
def vacances(date):
    if (year(date)==2011):
        if (month(date)==1 and day(date)<=3):
            return 1
        if (month(date)==2 and day(date)>=12):
            return 1
        if (month(date)==3 and day(date)<=14):
            return 1
        if (month(date)==4 and day(date)>=9):
            return 1
        if (month(date)==5 and day(date)<=9):
            return 1
        if ((month(date)>=7 and day(date)>=3) or month(date)==8):
            return 1
        if (month(date)==10 and day(date)>=22):
            return 1
        if (month(date)==11 and day(date)<=3):
            return 1
        if (month(date)==12 and day(date)>=17):
            return 1
    if (year(date)==2012):
        if (month(date)==1 and day(date)<=3):
            return 1
        if (month(date)==2 and day(date)>=11):
            return 1
        if (month(date)==3 and day(date)<=12):
            return 1
        if (month(date)==4 and day(date)>=7):
            return 1
        if (month(date)==5 and day(date)<=7):
            return 1
        if ((month(date)>=7 and day(date)>=5) or month(date)==8):
            return 1
        if (month(date)==10 and day(date)>=27):
            return 1
        if (month(date)==11 and day(date)<=8):
            return 1
        if (month(date)==12 and day(date)>=22):
            return 1
    if (year(date)==2013):
        if (month(date)==1 and day(date)<=7):
            return 1
        if (month(date)==2 and day(date)>=16):
            return 1
        if (month(date)==3 and day(date)<=18):
            return 1
        if (month(date)==4 and day(date)>=13):
            return 1
        if (month(date)==5 and day(date)<=13):
            return 1
        if ((month(date)>=7 and day(date)>=4) or month(date)==8):
            return 1
        if (month(date)==10 and day(date)>=19):
            return 1
        if (month(date)==11 and day(date)<=4):
            return 1
        if (month(date)==12 and day(date)>=21):
            return 1
    return 0
            

def saison(date):
    if (month(date)<3 or (month(date)==3 and day(date)<=21)):
        return 1
    if ((month(date)>3 and month(date)<6) or (month(date)==3 and day(date)>=22) or (month(date)==6 and day(date)<=21)):
        return 2
    if ((month(date)>6 and month(date)<9) or (month(date)==6 and day(date)>=22) or (month(date)==9 and day(date)<=21)):
        return 3
    if ((month(date)>9 and month(date)<12) or (month(date)==9 and day(date)>=22) or (month(date)==12 and day(date)<=21)):
        return 4
    if (month(date)==12 and day(date)>=22):
        return 1
    return 0
        



submission = pd.read_csv('submission.txt' ,sep='\t',parse_dates=[0])




test = submission.copy()
print("importing train")


my_data = pd.read_csv('Train_preproc.csv',delimiter=';')
print("fixing test data")
all_ass = np.unique(submission['ASS_ASSIGNMENT'].values)        
print(all_ass.size)
# we add the missing columns to the testing set
test['month']=(pd.to_datetime(test['DATE'])).map(month)
test['day']=(pd.to_datetime(test['DATE'])).map(day)
test['weekday']=(pd.to_datetime(test['DATE'])).map(weekday)
test['weekend']=(pd.to_datetime(test['DATE'])).map(weekend)
test['time']=(pd.to_datetime(test['DATE'])).map(time)
test['night']=(pd.to_datetime(test['DATE'])).map(night)
test['year']=(pd.to_datetime(test['DATE'])).map(year)
test['jours_ferie']=(pd.to_datetime(test['DATE'])).map(jours_ferie)
test.drop('DATE' ,axis=1,inplace=True) # axis=1 => we delete a column, inplace=True => we don't need to create a new dataframe ( high cost in memory)
test.drop('prediction' ,axis=1,inplace=True)
print("preprocessing")

#creating dummies for ASS
for ass in all_ass :
       
        
    test["ASS_"+ass] = (test['ASS_ASSIGNMENT'] == ass)*1
    my_data["ASS_"+ass] = (my_data['ASS_ASSIGNMENT'] == ass)*1





weather=  pd.read_csv('weather.csv',delimiter=';')  

# the we don't take all the features of weather to make the learning faster
#the following columns were chosen using the " feature_importances_ " of sklearn randomForest estimator
weather=weather[[ 'month','day','year','Temp\xe9rature maximumC',
       'Temp\xe9rature minimumC', 'Point de ros\xe9eC', 'MeanDew PointC',
       'Min DewpointC', ' Min Humidit\xe9',
       ' Max Pression au niveau de la merhPa',
       ' Mean Pression au niveau de la merhPa',
       ' Min Pression au niveau de la merhPa', ' Mean Visibilit\xe9Km',
       ' Max Vitesse du ventKm/h', ' Mean Vitesse du ventKm/h',
       'Pr\xe9cipitationmm', ' CloudCover', 'WindDirDegrees', 'Orage']]
my_data=my_data.merge(weather)
test=test.merge(weather)

my_data.drop('ASS_ASSIGNMENT' ,axis=1,inplace=True)
test.drop('ASS_ASSIGNMENT' ,axis=1,inplace=True)


# the linex function used to evaluate our error on training and evaluations sets
def score_func_linex(y, y_pred):
    
    
    s=np.exp(0.1*(y-y_pred))-(0.1)*(y-y_pred)-1
    
    return s.sum()

# the scorer, used for sklearn crossvalidation, not used here
linexscorer=make_scorer(score_func_linex,greater_is_better=False)




print("working")

#importing xgboost
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

# the follwing are custom objective function that we tried with xgboost, but we ended up using a pre-defined one
 
def custom(y, y_pred):
    y=np.array(y)
    y_pred=np.array(y_pred)
    grad=np.zeros_like(y)
    hess=np.zeros_like(y)
    hess=hess+16
    grad=grad-16*(y-y_pred)
    bad= np.where(y>y_pred)
    grad[bad]=-16*(y[bad]-y_pred[bad])*(y[bad]-y_pred[bad])*(y[bad]-y_pred[bad])
    hess[bad]=16*(y[bad]-y_pred[bad])*(y[bad]-y_pred[bad])
    return grad,hess
    
    

def custom3(y, y_pred):
    y=np.array(y)
    y_pred=np.array(y_pred)
    grad=np.zeros_like(y)
    hess=np.zeros_like(y)
    hess=hess+2
    grad=grad-2*(y-y_pred)
    bad= np.where(y>y_pred)
    grad[bad]=-(y[bad]-y_pred[bad])*(y[bad]-y_pred[bad])*(y[bad]-y_pred[bad])
    hess[bad]=(y[bad]-y_pred[bad])*(y[bad]-y_pred[bad])
    return grad,hess

def custom2(y, y_pred):
    y=np.array(y)
    y_pred=np.array(y_pred)
    grad=np.zeros_like(y)
    hess=np.zeros_like(y)
    hess=hess+2
    grad=grad-2*(y-y_pred)
    bad= np.where(y>y_pred)
    grad[bad]=-16*(y[bad]-y_pred[bad])
    hess[bad]=16
    return grad,hess
 



clf = xgb.XGBRegressor(max_depth=20, n_estimators=300,objective= "reg:linear", silent =False,min_child_weight=0.01,learning_rate=0.1,subsample = 0.8, colsample_bytree = 0.8)
# fitting and predicting using data transformation for the number of recieved calls
my_data['date']=pd.to_datetime(my_data.year*10000+my_data.month*100+my_data.day,format='%Y%m%d')
test['date']=pd.to_datetime(test.year*10000+test.month*100+test.day,format='%Y%m%d')





test=test.sort_values('date')

# we seperate the target values  and the features
y=my_data['CSPL_RECEIVED_CALLS']

my_data.drop('CSPL_RECEIVED_CALLS' ,axis=1,inplace=True)


longueur= test.shape[0]
# the predicted values
prediction=np.zeros((longueur))

#we store the dates at which we should stop training for every predicted week
beginning_of_predicted_weeks={}
beginning_of_predicted_weeks[0]='2012-12-28'
beginning_of_predicted_weeks[1]='2013-02-02'
beginning_of_predicted_weeks[2]='2013-03-06'
beginning_of_predicted_weeks[3]='2013-04-10'
beginning_of_predicted_weeks[4]='2013-05-13'
beginning_of_predicted_weeks[5]='2013-06-12'
beginning_of_predicted_weeks[6]='2013-07-16'
beginning_of_predicted_weeks[7]='2013-08-15'
beginning_of_predicted_weeks[8]='2013-09-14'
beginning_of_predicted_weeks[9]='2013-10-18'
beginning_of_predicted_weeks[10]='2013-11-20'
beginning_of_predicted_weeks[11]='2013-12-22'
# last value is for practical use of the code : it just need to be after the last predicted date ( meaning after 2013-12-28)
beginning_of_predicted_weeks[12]='2014-01-01'


for i in range(12):
    testing_indexes=np.where((test['date']>= beginning_of_predicted_weeks[i]) * (test['date']< beginning_of_predicted_weeks[i+1])) 
    s=test.iloc[testing_indexes ]
    
    s=s.drop('date' ,axis=1)
    my_data2=my_data[my_data['date']< beginning_of_predicted_weeks[i]].drop('date' ,axis=1)
    y2=y[my_data['date']<beginning_of_predicted_weeks[i]]
    #training with reieved calls transformation
    clf.fit(my_data2,np.exp(0.0001*y2))
    #prediction with reieved calls inverse transformation
    prediction[testing_indexes]=np.log(clf.predict(s))*(1/0.0001)
    
    del(my_data2)



print("predicting")
submission.prediction=prediction*1.09
print("exporting")
submission.prediction=submission.prediction.clip(lower = 0)
submission['prediction']=submission['prediction'].apply(round)
submission.to_csv('gg_xgboost_300est.txt',sep='\t',date_format='%Y-%m-%d %H:%M:%S.000',index=False ,encoding='utf-8')

# finally , we see the execution time
time_final = t.time()
print(time_final - start)






