# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:01:54 2020

@author: XBSD
"""

import talib as ta
import numpy as np
import baostock as bs
import pandas as pd
import pickle
import seaborn as sns
#from sklearn import 益率；


from sklearn.ensemble import RandomForestClassifier as RM  
import sklearn.metrics as me
#%%

result=pd.read_csv('benchmark_timing_sz50.csv')
df=result.set_index('date')
df['volume']=df['volume'].map(lambda x:float(x))

#%%

close = df['close'].values
volume = df['volume'].values
high = df['high'].values
low = df['low'].values
print(df.head())
#%%
adx = ta.ADX(high, low, close, timeperiod=14).tolist()
rsi = ta.RSI(close, timeperiod=14).tolist()
mfi = ta.MFI(high, low, close, volume, timeperiod=14).tolist()
momentum = ta.MOM(close, timeperiod=10).tolist()
var = ta.VAR(close, timeperiod=5, nbdev=1).tolist()
aroonosc = ta.AROONOSC(high, low, timeperiod=14).tolist()
ema = ta.EMA(close, timeperiod=30).tolist()
linreg = ta.LINEARREG(close, timeperiod=14).tolist()
cycle = ta.HT_DCPERIOD(close).tolist()
atr = ta.ATR(high, low, close, timeperiod=14).tolist()
#%%
X = np.array([adx,rsi,mfi,momentum,var,aroonosc,ema,linreg,cycle,atr]).T
data_x=pd.DataFrame(X[35:],columns=['adx','rsi','mfi','momentum','var','aroonosc','ema','linreg','cycle','atr'])
print(data_x.head())

#%%
labels=np.log(df['close']/df['close'].shift(5))
labels = labels.dropna()
print(labels.head())
#%%
threshold = 0.01 
y1 = np.where(labels.values >= threshold, 1, 0) 
y = y1[labels.shape[0] - data_x.shape[0]:]
#%%
seperator=.6 
length=len(y)
X_train, X_test = data_x[:int(length * seperator)], data_x[int(length * seperator):]
y_train, y_test = y[:int(length * seperator)], y[int(length * seperator):]
print (u'训练集共有数据%d个'%len(X_train))
print (u'测试集共有数据%d个'%len(X_test))

#%%
model = RM().fit(X_train, y_train) 
y_pred = model.predict(X_test) 
print (me.classification_report(y_test,y_pred)) 

#%%
important_features=model.feature_importances_
feature_impDF = pd.DataFrame()
feature_impDF["importance"] =important_features
feature_impDF["feature"] = data_x.columns

feature_impDF.sort_values("importance",ascending=False,inplace=True)
feature_impDF.set_index(feature_impDF.feature,inplace=True)
feature_impDF.importance.plot(kind="bar")
feature_impDF

#%%
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

from sklearn import preprocessing

#划分训练，测试集
experData_X=X_train
#experData_X=train[['change',"volume"]]
experData_y=y_train

experData_X=preprocessing.scale(experData_X)
#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)

#汇总不同模型算法
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
#classifiers.append(AdaBoostClassifier())




        
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))

#求出模型得分的均值和标准差
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
#汇总数据
cvResDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})

print(cvResDf)
sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})


#GradientBoostingClassifier模型
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(experData_X,experData_y)

#LogisticRegression模型
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold, 
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_y)
#modelgsGBC模型
print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
#modelgsLR模型
print('modelgsLR模型得分为：%.3f'%modelgsLR.best_score_)


#查看模型ROC曲线
#求出测试数据模型的预测值
modelgsGBCtestpre_y=modelgsGBC.predict(experData_X).astype(int)
#画图
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(experData_y, modelgsGBCtestpre_y) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Titanic GradientBoostingClassifier Model')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import confusion_matrix
print('GradientBoostingClassifier模型混淆矩阵为\n',confusion_matrix(experData_y.astype(int).astype(str),modelgsGBCtestpre_y.astype(str)))
print('LinearRegression模型混淆矩阵为\n',confusion_matrix(experData_y.astype(int).astype(str),modelgsGBCtestpre_y.astype(str)))




















