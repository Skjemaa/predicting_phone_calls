import numpy as np
import pandas as pd
import datetime


start = t.time()
submission = pd.read_csv('submission.txt',dtype={'DATE':'category','ASS_ASSIGNMENT':'category','prediction': np.int16,} ,sep='\t',parse_dates=[0])


all_ass = np.unique(submission['ASS_ASSIGNMENT'].values)        

n=all_ass.size

# finally we cancat the data from the differents ASS so we can use them for the training       
del(submission)
ass_list=list()
for ass in all_ass :
     ass_sep = pd.read_csv('ASS='+ass+'clean_merged.csv', sep=';')
     ass_list.append(ass_sep)
     
result=pd.concat([ass_list[i] for i in range(n)])
result.to_csv('Train_preproc.csv', sep=';',index=False)