# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:19:22 2020

@author: Administrator
"""


import pandas as pd
import matplotlib.pylab as plt
from arch.unitroot import ADF
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.externals import joblib
import itertools
import scipy.stats as stats

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

class Model:
    
    def __init__(self, df, period, p_max, q_max, d_max, forecast_num):
        self.df = df
        self.period = period
        self.p_max = p_max
        self.q_max = q_max
        self.d_max = d_max
        self.forecast_num = forecast_num
    
    def diff_process(self):
        self.ADF_value = ADF(self.df.iloc[:,0]) #p值为0小于0.05认为是平稳的(单位根检验)
    
        self.diff_period = self.df.iloc[:,0].diff(self.period) #季节性差分
        self.diff_period = self.diff_period.dropna()
        self.diff_ = self.diff_period.diff() #一次差分
        self.diff_ = self.diff_.dropna()
        
        fig = plt.figure(figsize=(20,6))
        ax1 = fig.add_subplot(311) #原始数据图
        ax1.plot(self.df.iloc[:,0])
        ax2 = fig.add_subplot(312) #季节性查分差分后 无周期性 但是不平稳
        ax2.plot(self.diff_period)
        ax3 = fig.add_subplot(313) #再一次差分之后 平稳
        ax3.plot(self.diff_)
        plt.show()
        
    def acf_pacf_fig(self):
        
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.df.iloc[:,0], lags = 100, ax = ax1)
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.df.iloc[:,0], lags = 100, ax = ax2)
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
        # pq看季节性差分和差分之后的
        # PQ看季节性差分之后
        # 由ACF：滞后1和95即可，故qQ均为1
        # 由PACF：

        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(self.diff_period, lags = 200, ax = ax1)
        ax1.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(self.diff_period, lags = 200, ax = ax2)
        ax2.xaxis.set_ticks_position('bottom')
        fig.tight_layout()
        plt.show()
        # 由ACF：95处显著，190处不显著，故Q=1
        # 由PACF：95和190均显著，故P=1，2均可能
        
    def params_select(self):
        self.p = range(0, self.p_max)
        self.q = range(0, self.q_max)
        d = range(1, self.d_max)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(self.p, d, self.q))
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], self.period) for x in list(itertools.product(self.p, d, self.q))]

        aic_value = pd.DataFrame()
        for i, param in enumerate(pdq):
            for param_seasonal in seasonal_pdq:
                model = sm.tsa.statespace.SARIMAX(self.df.iloc[:,0],
                                                  order = param,
                                                  seasonal_order = param_seasonal,
                                                  enforce_stationarity = False,
                                                  enforce_invertibility = False)
                results = model.fit()
                print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                param_list = [[param, param_seasonal, results.aic]]
                aic_value_ = pd.DataFrame(param_list, columns = ['param', 'param_seasonal', 'aic'])
                aic_value = pd.concat([aic_value, aic_value_])
                    
        index_list = []
        for i in range(self.p_max * self.q_max * self.p_max * self.q_max):
            index_list.append(i)
        aic_value.index = index_list 
        
        min_index = aic_value[aic_value.aic==min(aic_value['aic'])].index #找到aic值最小的行索引
        a = aic_value.iloc[min_index,:]
        a = a.values.tolist()
        self.param = a[0][0]
        self.param_seasonal = a[0][1]

    def sarima(self):
        model = SARIMAX(self.df.iloc[:,0], order = self.param, seasonal_order = self.param_seasonal)#与上一句等价
        print('the best parameters: SARIMA{}x{}'.format(self.param, self.param_seasonal))
        self.results = model.fit()
        #joblib.dump(results, f'C:\\Users\\Administrator\\Desktop\\SARIMA模型.pkl')
        self.predict_ = self.results.forecast(self.forecast_num)

        fig, ax = plt.subplots(figsize=(30,6))
        ax = self.predict_.plot(ax = ax)
        self.df.iloc[:,0].plot(ax = ax)
        plt.legend(['y_pred', 'y_true'])
        plt.show()
        return self.results
    
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