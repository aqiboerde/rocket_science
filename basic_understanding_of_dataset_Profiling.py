# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 23:38:17 2020

@author: Lisa
"""
import pandas as pd
import pickle 
import numpy as np
import seaborn as sns

import pandas_profiling as pp
import matplotlib.pyplot as plt

file=open('sz50_stocks_list.pickle','rb')
list_50=pickle.load(file)
file.close()

stock=list_50[1]
data=pd.read_csv('johnston_dataset/with_2_rolling'+stock+'.csv')
report = pp.ProfileReport(data)
report.to_file('profile_report.html')
