import numpy as np
import pandas as pd
import datetime







submission = pd.read_csv('submission.txt',dtype={'DATE':'category','ASS_ASSIGNMENT':'category','prediction': np.int16,} ,sep='\t',parse_dates=[0])


all_ass = np.unique(submission['ASS_ASSIGNMENT'].values)        
print(all_ass.size)

        
del(submission)




# we load every file containing the data for every ASS. We notice that for the same time slot in the same day, we have different rows,
# these rows contains the number of reieved calls from different calls center, we have to make their sum to obtain the real number of recieved calls in that time slot

for ass in all_ass :
     ass_sep = pd.read_csv('ASS='+ass+'clean.csv', sep=';')
     
     era=ass_sep.groupby(['ASS_ASSIGNMENT','month','day','weekday','weekend','time','night','year','jours_ferie'])['CSPL_RECEIVED_CALLS'].sum()
     
     era=era.reset_index()
     era.to_csv('ASS='+ass+'clean_merged.csv', sep=';',index=False)

     
     
