# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:25:59 2020

@author: Administrator
"""


import pandas as pd
import matplotlib.pylab as plt
from TimeSeries.time_series_model import Model
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

period = 10
p_max = 2
q_max = 4
d_max = 2
forecast_num = 300
pdq_ = (5, 2, 4)
PDQ_ = (4, 2, 3, 10)
df = pd.read_excel('C:\\Users\\Administrator\\Desktop\\TimeSeries\\时序.xlsx')

My_model = Model(df, period, p_max, q_max, d_max, forecast_num, pdq_, PDQ_)
My_model.diff_process()
if My_model.p_value[1] > 0.05:
    print('此时间序列不具有相关性，ARIMA模型不适用')
else:
    My_model.acf_pacf_fig()
    try:
        My_model.params_select()
        My_model.sarima()
    except:
        print('\n', 'ERROR', '\n')
        My_model.sarima_()
    My_model.model_eval()
    My_model.qq_plot()
    
# My_model.acf_pacf_fig()或My_model.params_select()确定模型参数 