import baostock as bs
import pandas as pd
import pickle

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息

rs = bs.query_history_k_data_plus('sh.000016',
        "date,code,open,high,low,close,volume,amount,adjustflag",
        start_date='2010-01-01', end_date='2020-01-31',
        frequency="d", adjustflag="3")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)


#file=open('benchmark_sz50.pickle','wb')
#pickle.dump(result,file)
#file.close()