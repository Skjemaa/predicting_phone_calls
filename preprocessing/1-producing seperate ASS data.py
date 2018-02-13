import numpy as np
import pandas as pd
import datetime



# followings are functions we use to extract  features from the Date column

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
            
# following are data that we drop because we are not going to use        
droplist =[]
droplist.extend(['SPLIT_COD','CSPL_CALLSOFFERED','CSPL_OUTFLOWCALLS','CSPL_INFLOWCALLS','CSPL_NOANSREDIR','CSPL_ACDCALLS',
    'CSPL_ABNCALLS','CSPL_CONFERENCE','CSPL_TRANSFERED','CSPL_RINGCALLS','CSPL_DISCCALLS','CSPL_HOLDCALLS',
    'CSPL_ACDAUXOUTCALLS','CSPL_HOLDABNCALLS','CSPL_MAXINQUEUE','CSPL_DEQUECALLS','CSPL_ACWINCALLS',
    'CSPL_AUXINCALLS','CSPL_ACWOUTCALLS','CSPL_ACWOUTOFFCALLS','CSPL_ACWOUTADJCALLS','CSPL_AUXOUTCALLS',
    'CSPL_AUXOUTOFFCALLS','CSPL_AUXOUTADJCALLS','CSPL_INTRVL','CSPL_OUTFLOWTIME','CSPL_DEQUETIME','CSPL_I_ACDTIME',
    'CSPL_DISCTIME','CSPL_HOLDTIME','CSPL_ABNTIME','CSPL_I_STAFFTIME','CSPL_ANSTIME','CSPL_I_RINGTIME',
    'CSPL_RINGTIME','CSPL_ACDTIME','CSPL_I_AVAILTIME','CSPL_ACWTIME','CSPL_I_ACWTIME','CSPL_I_OTHERTIME',
    'CSPL_ACWINTIME','CSPL_I_ACWINTIME','CSPL_AUXINTIME','CSPL_I_AUXINTIME','CSPL_ACWOUTIME','CSPL_I_ACWOUTTIME',
    'CSPL_ACWOUTOFFTIME','CSPL_AUXOUTTIME','CSPL_I_AUXOUTTIME','CSPL_AUXOUTOFFTIME','CSPL_SERVICELEVEL',
    'CSPL_ACCEPTABLE','CSPL_SLVLOUTFLOWS','CSPL_SLVLABNS','CSPL_ABNCALLS1','CSPL_ABNCALLS2','CSPL_ABNCALLS3',
    'CSPL_ABNCALLS4','CSPL_ABNCALLS5','CSPL_ABNCALLS6','CSPL_ABNCALLS7','CSPL_ABNCALLS8','CSPL_ABNCALLS9',
    'CSPL_ABNCALLS10','CSPL_MAXSTAFFED','CSPL_INCOMPLETE','CSPL_ABANDONNED_CALLS','CSPL_CALLS'])
droplist.extend(['DAY_OFF','DAY_DS'])
droplist.extend(['ACD_COD','ACD_LIB'])
droplist.extend(['ASS_DIRECTORSHIP','ASS_PARTNER','ASS_POLE','ASS_SOC_MERE'])
droplist.extend(['ASS_BEGIN','ASS_COMENT','ASS_END'])
    
 # the list of ASS that don't exist in submission.txt, we are not going to use them for the training       
non_traite = ['A DEFINIR', 'AEVA', 'DOMISERVE', 'Divers','Evenements', 'FO Remboursement', 'Finances PCX',
       'IPA Belgique - E/A MAJ','Juridique', 'KPT', 'LifeStyle','Maroc - Génériques', 'Maroc - Renault',
       'Medicine', 'NL Médical', 'NL Technique','Réception', 'TAI - CARTES', 'TAI - PANNE MECANIQUE',
       'TAI - PNEUMATIQUES', 'TAI - RISQUE', 'TAI - RISQUE SERVICES','TAI - SERVICE', 'TPA',
       'Technical', 'Technique Belgique', 'Technique International','Truck Assistance']        

        
        
        
        
        
submission = pd.read_csv('submission.txt',dtype={'DATE':'category','ASS_ASSIGNMENT':'category','prediction': np.int16,} ,sep='\t',parse_dates=[0])


all_ass = np.unique(submission['ASS_ASSIGNMENT'].values)        
        
del(submission)








print("reading excel file")

# we import the data, and we precise the type for every column, using low memory types ( like numpy.int16 instead of numpy.int64 )to avoid memory 

my_data = pd.read_csv('train_2011_2012_2013.csv',delimiter=';',dtype={'DATE':'category','DAY_OFF' : np.int16,'DAY_DS':'category',
                                                                  'WEEK_END': np.int16,'DAY_WE_DS':'category','TPER_TEAM':'category','TPER_HOUR' : np.int16
                                                                  ,'SPLIT_COD': np.int16,'ACD_COD':np.int16,'ACD_LIB':'category','ASS_SOC_MERE':'category',
                                                                  'ASS_DIRECTORSHIP':'category','ASS_ASSIGNMENT':'category','ASS_PARTNER':'category','ASS_POLE':'category',
                                                                  'ASS_BEGIN':'category','ASS_END':'category','ASS_COMENT':'category','CSPL_I_STAFFTIME': np.int32,
                                                                  'CSPL_I_AVAILTIME': np.int32,'CSPL_I_ACDTIME': np.int32,'CSPL_I_ACWTIME': np.int32,'CSPL_I_ACWOUTTIME': np.int16
                                                                  ,'CSPL_I_ACWINTIME': np.int16,'CSPL_I_AUXOUTTIME': np.int16,'CSPL_I_AUXINTIME': np.int16,'CSPL_I_OTHERTIME': np.int32,
                                                                  'CSPL_ACWINCALLS': np.int16,'CSPL_ACWINTIME': np.int16,'CSPL_AUXINCALLS': np.int16,'CSPL_AUXINTIME': np.int16,
                                                                  'CSPL_ACWOUTCALLS': np.int16,'CSPL_ACWOUTIME': np.int16,'CSPL_ACWOUTOFFCALLS': np.int16,'CSPL_ACWOUTOFFTIME'
                                                                  : np.int16,'CSPL_AUXOUTCALLS': np.int16,'CSPL_AUXOUTTIME': np.int16,'CSPL_AUXOUTOFFCALLS': np.int16,
                                                                  'CSPL_AUXOUTOFFTIME': np.int16,'CSPL_INFLOWCALLS': np.int16,'CSPL_ACDCALLS': np.int16,'CSPL_ANSTIME': np.int32
                                                                  ,'CSPL_HOLDCALLS': np.int16,'CSPL_HOLDTIME': np.int16,'CSPL_HOLDABNCALLS': np.int16,'CSPL_TRANSFERED': np.int16,
                                                                  'CSPL_CONFERENCE': np.int16,'CSPL_ABNCALLS': np.int16,'CSPL_ABNTIME': np.int32,'CSPL_ABNCALLS1': np.int16,
                                                                  'CSPL_ABNCALLS2': np.int16,'CSPL_ABNCALLS3': np.int16,'CSPL_ABNCALLS4': np.int16,'CSPL_ABNCALLS5': np.int16,
                                                                  'CSPL_ABNCALLS6': np.int16,'CSPL_ABNCALLS7': np.int16,'CSPL_ABNCALLS8': np.int16,'CSPL_ABNCALLS9': np.int16,
                                                                  'CSPL_ABNCALLS10': np.int16,'CSPL_OUTFLOWCALLS': np.int16,'CSPL_OUTFLOWTIME': np.int16,'CSPL_MAXINQUEUE': np.int16
                                                                  ,'CSPL_CALLSOFFERED': np.int16,'CSPL_I_RINGTIME': np.int16,'CSPL_RINGTIME': np.int16,'CSPL_RINGCALLS': np.int16,
                                                                  'CSPL_NOANSREDIR': np.int16,'CSPL_MAXSTAFFED': np.int16,'CSPL_ACWOUTADJCALLS': np.int16,'CSPL_AUXOUTADJCALLS': np.int16,
                                                                  'CSPL_DEQUECALLS': np.int16,'CSPL_DEQUETIME': np.int32,'CSPL_DISCCALLS': np.int16,'CSPL_DISCTIME': np.int16,
                                                                  'CSPL_INTRVL': np.int16,'CSPL_INCOMPLETE': np.int16,'CSPL_ACCEPTABLE': np.int16,'SPL_SERVICELEVEL': np.int16,
                                                                  'CSPL_ACDAUXOUTCALLS': np.int16,'CSPL_SLVLABNS': np.int16,'CSPL_SLVLOUTFLOWS': np.int16,'CSPL_RECEIVED_CALLS': np.int16,
                                                                  'CSPL_ABANDONNED_CALLS': np.int16,'CSPL_CALLS': np.int16,'CSPL_ACWTIME': np.int32,'CSPL_ACDTIME': np.int32 })
# we apply the features extration from Date column
print("preprocessing")
my_data.drop(['DAY_WE_DS','TPER_TEAM','TPER_HOUR','WEEK_END'], axis=1,inplace=True  )  
my_data['month']=(pd.to_datetime(my_data['DATE'])).map(month)
#my_data['month'].value_counts()
my_data['day']=(pd.to_datetime(my_data['DATE'])).map(day)
my_data['weekday']=(pd.to_datetime(my_data['DATE'])).map(weekday)
my_data['weekend']=(pd.to_datetime(my_data['DATE'])).map(weekend)
my_data['time']=(pd.to_datetime(my_data['DATE'])).map(time)
my_data['night']=(pd.to_datetime(my_data['DATE'])).map(night)
my_data['year']=(pd.to_datetime(my_data['DATE'])).map(year)
my_data['jours_ferie']=(pd.to_datetime(my_data['DATE'])).map(jours_ferie)
# axis=1 => we delete a column, inplace=True => we don't need to create a new dataframe ( high cost in memory)
my_data.drop('DATE' ,axis=1,inplace=True) 




my_data.drop(droplist, axis=1,inplace=True)

# we remove the ASS that we are not going to use
print("removing extra")
for ass in non_traite:
        my_data = my_data[my_data['ASS_ASSIGNMENT']!=ass]

# finally , we create a seperate file for every ASS, this will help us visualize the data 
print("creating ass")
for ass in all_ass :
     ass_sep=   (my_data[my_data['ASS_ASSIGNMENT']==ass] ).copy()
     ass_sep.to_csv('ASS='+ass+'clean.csv', sep=';',index=False)
        
    
    
