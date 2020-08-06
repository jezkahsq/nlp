# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:28:48 2020

@author: Administrator
"""


import pandas as pd
import matplotlib.pylab as plt
from TimeSeries.time_series_model import Model
from TimeSeries.time_series_arima import Arima

import warnings
warnings.filterwarnings("ignore") 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

# 输入参数
p_max = 6
q_max = 6
forecast_num = 100

# 读取时序数据
df = pd.read_excel('C:\\Users\\Administrator\\Desktop\\TimeSeries\\6月均值.xlsx')

My_model = Arima(df, p_max, q_max, forecast_num)
My_model.diff_process()
if My_model.p_value[1] > 0.05:
    print('此时间序列不具有相关性，ARIMA模型不适用')
else:
    My_model.acf_pacf_fig()
    My_model.params_select()

# 训练模型及预测
My_model.arima()

# 模型评估
My_model.model_eval()
My_model.qq_plot()