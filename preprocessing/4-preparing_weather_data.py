import numpy as np
import pandas as pd
import datetime

import time as t
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn import preprocessing

start = t.time()
# concating weather data for 3 years
weather1=pd.read_csv('meteo_2011.csv',sep=',',parse_dates=[0])
weather2=pd.read_csv('meteo_2012.csv' ,sep=',',parse_dates=[0])
weather3=pd.read_csv('meteo_2013.csv' ,sep=',',parse_dates=[0])



weather=pd.concat([weather1,weather2,weather3])
#☻ we drop this columns because of too many missing values
weather.drop(' Max Vitesse des rafalesKm/h' ,axis=1,inplace=True)


#creating categories dummies for evennement
weather['Pluie']= 0
weather['Brouillard']= 0
weather['Orage']= 0
weather['Neige']= 0
n=weather.shape[0]
weather=weather.reset_index()
# Creating dummies from Evenement ( knowing that one evenement can lead to more than one positive value of dummies)
for i in range(n):  
    
    
    if ( isinstance( weather.iloc[i][' Ev\xe9nements'],float ) and   np.isnan(weather.iloc[i][' Ev\xe9nements'])):
        a=False
    else :
        a= 'Pluie'in weather.iloc[i][' Ev\xe9nements']
    weather.set_value(i,'Pluie',a*1)
    
    if ( isinstance( weather.iloc[i][' Ev\xe9nements'],float ) and   np.isnan(weather.iloc[i][' Ev\xe9nements'])):
        a=False;
    else:
        a= 'Brouillard'in weather.iloc[i][' Ev\xe9nements']
    weather.set_value(i,'Brouillard',a*1)
    
    if ( isinstance( weather.iloc[i][' Ev\xe9nements'],float ) and   np.isnan(weather.iloc[i][' Ev\xe9nements'])):
        a=False;
    else:
        a= 'Orage'in weather.iloc[i][' Ev\xe9nements']
    weather.set_value(i,'Orage',a*1)
    
    if ( isinstance( weather.iloc[i][' Ev\xe9nements'],float ) and   np.isnan(weather.iloc[i][' Ev\xe9nements'])):
        a=False;
    else:
        a= 'Neige'in weather.iloc[i][' Ev\xe9nements']
    weather.set_value(i,'Neige',a*1)

weather.drop(' Ev\xe9nements' ,axis=1,inplace=True)


#fillNA



weather.fillna(axis=0,method='ffill', inplace=True)
weather.drop('index' ,axis=1,inplace=True)
#getting date so we can merge easily with our data

def year(date) :
    return date.year
def month(date) :
    return date.month
    
def day(date) : 
    return date.day
weather['month']=(pd.to_datetime(weather['CET'])).map(month)
weather['day']=(pd.to_datetime(weather['CET'])).map(day)
weather['year']=(pd.to_datetime(weather['CET'])).map(year)

weather.drop('CET' ,axis=1,inplace=True)


#exporting result

weather.to_csv('weather.csv', sep=';',index=False)


























