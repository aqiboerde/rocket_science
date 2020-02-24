# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 00:14:10 2020

@author: Lisa
"""
import pandas as pd
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%%产生用于回测的基础数据，两张表记录价格信息和交易信号

file=open('sz50_stocks_list.pickle','rb')
list_50=pickle.load(file)
file.close()

dict={}
for stock in list_50: 
    try:
        data=pd.read_csv('johnston_dataset/with_2_rolling'+stock+'.csv')
        data.set_index(["date"], inplace=True)
        closeprice = data.close
        dict[stock] = closeprice
    except:
        dict[stock] = 0
        pass
df_price=pd.DataFrame(dict)
df_price.to_csv('closepricegraph.csv')

'''get+-pcmdatacsv'''
dict={}
for stock in list_50:
    try:
        data=pd.read_csv('johnston_dataset/with_2_rolling'+stock+".csv",index_col=0)
        rolling5_1430 = data['1430_rolling(5)']
        dict[stock] = rolling5_1430
    except:
        dict[stock] = 0
        pass
print(dict)
df_rolling=pd.DataFrame(dict)
df_rolling.to_csv('bigorder3.csv')


#%%
#def test(a,b):
#a为持股个数，b为持股天数
stockneeddict={}
numofstockchoose=15
numofdays_hold=1
for i in df_rolling.index:
    stocklist=df_rolling.sort_values(axis=1,by=i,ascending=False).loc[i].index.tolist()[
                     :numofstockchoose]
    stockneeddict[i]=stocklist

df=pd.DataFrame(stockneeddict)
df=df[df.columns.tolist()[0:-1:numofdays_hold]]
#%% 回测主体，在买卖时可以添加手续费，滑点等指标
date_list=df.columns.tolist()
for date_index,date in enumerate(date_list):
    for i in list(range(numofstockchoose,2*numofstockchoose)):
        if date_index==0:
            df.loc[i,date]=100
        else:
            df.loc[i,date]=df_price.loc[date_list[date_index],stockneeddict[date_list[date_index-1]][i-numofstockchoose]]
    for i in list(range(2*numofstockchoose,3*numofstockchoose)):
        df.loc[i,date]=df_price.loc[date,stockneeddict[date][i-2*numofstockchoose]]

for date_index,date in enumerate(date_list):
    for i in list(range(3*numofstockchoose,4*numofstockchoose)):
        if date_index==0:
            df.loc[i,date]=df.loc[i-2*numofstockchoose,date]/df.loc[i-1*numofstockchoose,date]
        else:
            df.loc[i,date]=df.loc[i,date_list[date_index-1]]*df.loc[i-2*numofstockchoose,date]/df.loc[i-1*numofstockchoose,date]
    df.loc[i+1,date]=sum(np.array(df[3*numofstockchoose:4*numofstockchoose][date])*np.array(df[2*numofstockchoose:3*numofstockchoose][date]))
#%% 回测数据分析，业绩指标计算
file=open('benchmark_sz50.pickle','rb')
benchmark=pickle.load(file)
file.close()
benchmark=benchmark.set_index('date')
benchmark['net_value']=0
for date_index,date in enumerate(date_list):
    benchmark.loc[date,'net_value']=df.loc[4*numofstockchoose,date]
benchmark=benchmark[benchmark['net_value']!=0]    
benchmark.rename(columns={'close':'benchmark', 'net_value':'account'}, inplace = True)
benchmark['benchmark']=benchmark['benchmark'].map(lambda x:float(x))

file=open('test_pefermance.pickle','wb')
pickle.dump(benchmark,file)
file.close()

benchmark['return_account']=np.log(benchmark['account'] / benchmark['account'].shift(1)) 
benchmark['return_benchmark']=np.log(benchmark['benchmark'] / benchmark['benchmark'].shift(1))
benchmark=benchmark.fillna(0)

risk_free=0
total_r_B=benchmark['benchmark'][-1]/benchmark['benchmark'][0]-1
total_r_A=benchmark['account'][-1]/benchmark['account'][0]-1
(r_A,r_B,sigma_A,sigma_B)=(benchmark['return_account'].mean(),benchmark['return_benchmark'].mean(),benchmark['return_account'].std(),benchmark['return_benchmark'].std())
annuralized_r_A=np.exp(r_A*len(benchmark)/5)-1
annuralized_r_B=np.exp(r_B*len(benchmark)/5)-1
annuralized_sigma_A=(250/numofdays_hold)**0.5*sigma_A
annuralized_sigma_B=(250/numofdays_hold)**0.5*sigma_B

sharp_account=(annuralized_r_A-risk_free)/annuralized_sigma_A
sharp_benchmark=(annuralized_r_B-risk_free)/annuralized_sigma_B

print("total return of account is %f, the benchmark is %f" % (total_r_A,total_r_B))
print("annualized return of account is %f, the benchmark is %f" % (annuralized_r_A,annuralized_r_B))
print("Annuralized sigma of account is %f, the benchmark is %f" % (annuralized_sigma_A,annuralized_sigma_B))

print('the sharp ratio of portfolio return is %f' % sharp_account )
print('the sharp ratio of benchmark return is %f' % sharp_benchmark)
data=np.array(benchmark['account'])
index_j = np.argmax(np.maximum.accumulate(data) - data)  # 结束位置
index_i = np.argmax(data[:index_j])  # 开始位置
d = data[index_j] - data[index_i]  # 最大回撤
maximum_drawdown=d/data[index_i]
print('protfolio Maximum drawdown is %f' % (maximum_drawdown))
benchmark['exces_return']=benchmark['return_account']-benchmark['return_benchmark']
t_stat=(r_A-0)/(sigma_A/np.sqrt(len(benchmark)))
print('the t-Stat that account differ from benchmark is %f' % (t_stat))

#    return[total_r_A,annuralized_r_A,annuralized_sigma_A,sharp_account,maximum_drawdown,t_stat]
    
#%% 画图
## 记得设置x轴格式
tick_spacing=100
fig, ax = plt.subplots(1,1)
ax.plot(benchmark.index,benchmark['benchmark']/benchmark['benchmark'][0],label='benchmark')
ax.plot(benchmark.index,benchmark['account']/benchmark['account'][0],label='account')
ax.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.show()

##%%样本内优化sharpe 基于持仓天数和持股个数
#day_interval=list(range(1,10))+list(range(10,65,5))
#numofstocks_list=list(range(1,10))+list(range(10,25,5))
#pefermance_outcome={}
#for a in day_interval[:]:
#    for b in numofstocks_list[:]:
#        pefermance_outcome[(a,b)]=test(a,b)
#
#df_pefermance_outcome=pd.DataFrame(pefermance_outcome)
##给出夏普比最大的参数组合，留意计算夏普比均为年化后的结果
#df_pefermance_outcome.sort_values(axis=1,by=3,ascending=False)
#file=open('pefermance_outcome.pickle','wb')
#
#pickle.dump(df_pefermance_outcome,file)
#file.close()
#
#
file=open('pefermance_outcome.pickle','rb')
df_pefermance_outcome=pickle.load(file)


















