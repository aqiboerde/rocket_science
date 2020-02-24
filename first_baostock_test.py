import baostock as bs
import pandas as pd
import pickle as pc
#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)



file=open('sz50_stocks_list.pickle','rb')
list_50=pc.load(file)

for i in list_50:

    rs = bs.query_history_k_data_plus(i,
        "date,preclose,code,open,high,low,close,volume,amount,adjustflag",
        start_date='2015-01-01', end_date='2020-01-31',
        frequency="d", adjustflag="3")
    # print('query_history_k_data_plus respond error_code:'+rs.error_code)
    # print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

#### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

#### 结果集输出到csv文件 ####   
    result.to_csv("sz50_hist_d/"+i+".csv", index=False)
    print(result.head())
    print(i)

#### 登出系统 ####
bs.logout()



















