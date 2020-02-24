# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 23:15:40 2020

@author: Lisa
"""
import pandas as pd
import pickle 
import numpy as np

file=open('sz50_stocks_list.pickle','rb')
list_50=pickle.load(file)
file.close()

for stock in list_50:

    df_d=pd.read_csv('sz50_hist_d/'+stock+'.csv',index_col='date')
    df_5min=pd.read_csv('sz50_hist_5min/'+stock+'.csv')
    
    df_5min['time']=df_5min['time'].map(lambda x: str(x)[8:12])
    df_d['1430_price']=df_5min[(df_5min['time']=='1430')].set_index('date').close
    df_d['1400_price']=df_5min[(df_5min['time']=='1400')].set_index('date').close
    
    df_d['1430_return']=np.log(df_d['1430_price']/df_d['1430_price'].shift(1))
    df_d['1400_return']=np.log(df_d['1400_price']/df_d['1400_price'].shift(1))
    
    df_d['close_return']=np.log(df_d['close']/df_d['close'].shift(1))
    
    df_d['1430_return_diff']=df_d['1430_return']-df_d['close_return']
    df_d['1400_return_diff']=df_d['1400_return']-df_d['close_return']
    
    df_d['1430_rolling(5)']=df_d['1430_return_diff'].rolling(5).sum()
    df_d['1400_rolling(5)']=df_d['1400_return_diff'].rolling(5).sum()
    
    df_d['highlow_return']=np.log(df_d['high']/df_d['low'])
    df_d['hl_diff']=df_d['highlow_return']-df_d['close_return']
    df_d['hl_rolling(5)']=df_d['hl_diff'].rolling(5).sum()
    
    df_d=df_d[5:]
    
    df_d.to_csv('johnston_dataset/with_2_rolling'+stock+'.csv')
    print(df_d.head())
    print(stock)
