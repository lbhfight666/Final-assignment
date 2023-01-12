# Final-assignment
## 编程基础期末大作业 李冰惠 2022103817

作业简介：二手车市场作为一个正在蓬勃发展且充满潜力的交易市场，越来越多的人将目光投向这里，作为爱好收藏，或是满足低价购车的需求， 抑或是相应资源再利用的政策号召，二手车市场都具备其存在的必要性。目前，二手车市场的定价往往是通过认为经验进行定价，缺乏统一 的标准和依据，故标准的定价方式不但是对市场参与者的重要参考依据，也是二手车市场可以持续健康发展的重要部分。本文希望通过统计方法构建模型，对二手车的价格进行预测。

运行环境： python3

使用库： numpy pandas os xgboost matplotlib seaborn missingno scipy warnings sklearn 

复现步骤： 按照代码顺序依次运行即可，数据均在附件
主要步骤：
（0）库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import scipy.stats as st
import warnings
（1）数据导入
train_data = pd.read_csv("car_train.csv", sep = " ")
test_data = pd.read_csv("car_test.csv", sep = " ")
（2）缺失&异常值处理
/缺失值
train_data['notRepairedDamage'].value_counts()
train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)

（3）探索性数据分析
（4）特征工程
（5）模型

