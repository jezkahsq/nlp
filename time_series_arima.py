# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:13:58 2020

@author: Administrator
"""


import pandas as pd
import matplotlib.pylab as plt
from arch.unitroot import ADF
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.externals import joblib
import scipy.stats as stats
import statsmodels.tsa.stattools as st  
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

class Arima:
    
    def __init__(self, df, p_max, q_max, forecast_num):
        self.df = df
        self.p_max = p_max
        self.q_max = q_max
        self.forecast_num = forecast_num
    
    def diff_process(self):
        
        self.p_value = acorr_ljungbox(self.df.iloc[:,1], lags=1) 
        print ('白噪声检验p值：', self.p_value[1], '\n') #大于0.05认为是白噪声，即序列在时间上不具有相关性
        #self.ADF_value = ADF(self.df.iloc[:,0]) #p值为0小于0.05认为是平稳的(单位根检验)
        '''
        单位根检验按p值判断是否平稳，否则一直作差分直到序列平稳
        '''
        self.diff_ = self.df.iloc[:,1]
        self.ADF_value = adfuller(self.diff_, autolag='AIC')
        self.i = 0
        while self.ADF_value[1] >= 0.05:
            self.diff_ = self.diff_.diff() #一次差分
            self.diff_ = self.diff_.dropna()
            self.ADF_value = adfuller(self.diff_, autolag='AIC') 
            # 1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较,
            # ADF Test result同时小于1%、5%、10%说明非常好的拒绝原假设，p值小于0.05，则平稳
            print('ADF检验:', '\n', self.ADF_value, '\n')
            self.i += 1
        
        fig = plt.figure(figsize=(20,6))
        ax1 = fig.add_subplot(211) #原始数据图
        ax1.plot(self.df.iloc[:,1])
        ax2 = fig.add_subplot(212) #再一次差分之后 平稳
        ax2.plot(self.diff_)
        plt.show()
        
    def acf_pacf_fig(self):
        
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.df.iloc[:,1], lags = 100, ax = ax1)
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.df.iloc[:,1], lags = 100, ax = ax2)
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        plt.show()
        
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.diff_, lags = 100, ax = ax1)
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.diff_, lags = 100, ax = ax2)
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        plt.show()


    def params_select(self):      
        self.order = st.arma_order_select_ic(self.diff_, max_ar = self.p_max, max_ma = self.q_max, ic=['aic', 'bic', 'hqic'])
        '''        
        常用的是AIC准则，AIC鼓励数据拟合的优良性但是尽量避免出现过度拟合(Overfitting)的情况。所以优先考虑的模型应是AIC值最小的那一个模型。        
        为了控制计算量，限制AR最大阶和MA最大阶，但是这样带来的坏处是可能为局部最优
        order.bic_min_order返回以BIC准则确定的阶数，是一个tuple类型
        '''
        self.order.bic_min_order = list(self.order.bic_min_order)
        self.order.bic_min_order.insert(1, self.i)
        self.order.bic_min_order = tuple(self.order.bic_min_order)
        print('the best parameters: ARIMA{}'.format(self.order.bic_min_order))

    def arima(self):
        model = ARIMA(self.df.iloc[:,1], order = self.order.bic_min_order)
        self.results = model.fit()
        #joblib.dump(results, f'C:\\Users\\Administrator\\Desktop\\ARIMA模型.pkl')
        self.predict_ = self.results.forecast(self.forecast_num)

        fig, ax = plt.subplots(figsize=(30,6))
        predict_and_df = np.concatenate((np.array(self.df.iloc[:,1]), self.predict_[0]))
        dt = {'x':[], 'y':[]}
        for i in range(len(predict_and_df)):
            dt['x'].append(i)
            dt['y'].append(predict_and_df[i])
            
        plot_dt = pd.DataFrame(dt, columns = ['x', 'y'])
        plt.plot(plot_dt.x[:len(self.df.iloc[:,1])], plot_dt.y[:len(self.df.iloc[:,1])], color = 'red')
        plt.plot(plot_dt.x[len(self.df.iloc[:,1]):], plot_dt.y[len(self.df.iloc[:,1]):], color = 'green')   
        plt.legend(['y_pred', 'y_true'])
        plt.show()
    
    def model_eval(self):
        #计算残差
        self.resid = self.results.resid
        
        #模型检验
        #残差的acf和pacf
        fig = plt.figure(figsize = (12,8))
        ax1 = fig.add_subplot(311)
        fig = sm.graphics.tsa.plot_acf(self.resid.values.squeeze(), lags = 40, ax = ax1) #squeeze()数组变为1维
        ax2 = fig.add_subplot(312)
        fig = sm.graphics.tsa.plot_pacf(self.resid, lags=40, ax = ax2)
        #残差自相关图断尾，所以残差序列为白噪声
      
    def qq_plot(self):
        plt.figure()
        stats.probplot(self.resid, dist="norm", plot=plt)
        plt.show()
        print('DW_value:', sm.stats.durbin_watson(self.resid))#DW值接近于２时，说明残差不存在（一阶）自相关性