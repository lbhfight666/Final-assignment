# Final-assignment
## 编程基础期末大作业 李冰惠 2022103817

作业简介：二手车市场作为一个正在蓬勃发展且充满潜力的交易市场，越来越多的人将目光投向这里，作为爱好收藏，或是满足低价购车的需求， 抑或是相应资源再利用的政策号召，二手车市场都具备其存在的必要性。目前，二手车市场的定价往往是通过认为经验进行定价，缺乏统一 的标准和依据，故标准的定价方式不但是对市场参与者的重要参考依据，也是二手车市场可以持续健康发展的重要部分。本文希望通过统计方法构建模型，对二手车的价格进行预测。

运行环境： python3

使用库： numpy pandas os xgboost matplotlib seaborn missingno scipy warnings sklearn 

复现步骤： 按照代码顺序依次运行即可，数据均在附件

主要步骤：<br>
（0）库<br>
import os<br>
import numpy as np<br>
import pandas as pd<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
import missingno as msno<br>
import scipy.stats as st<br>
import warnings<br>
（1）数据导入<br>
train_data = pd.read_csv("car_train.csv", sep = " ")<br>
test_data = pd.read_csv("car_test.csv", sep = " ")<br>
（2）缺失&异常值处理<br>
/缺失值<br>
train_data['notRepairedDamage'].value_counts()<br>
train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)<br>
test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)<br>
/缺失值可视化<br>
msno.matrix(train_data.sample(250)) <br>
msno.matrix(test_data.sample(250)) <br>
/异常值处理<br>
print("seller: ", train_data["seller"].value_counts())<br>
print("offerType:", train_data["offerType"].value_counts())<br>
del train_data["seller"]<br>
del train_data["offerType"]<br>
del test_data["seller"]<br>
del test_data["offerType"]<br>

def outliers_proc(data, col_name, scale=3): <br>
    """<br>
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗<br>
    :param data: 接收 pandas 数据格式<br>
    :param col_name: pandas 列名<br>
    :param scale: 尺度<br>
    :return: <br>
    """<br>
    def box_plot_outliers(data_ser, box_scale): <br>
        """<br>
        利用箱线图去除异常值<br>
        :param data_ser: 接收 pandas.Series 数据格式<br>
        :param box_scale: 箱线图尺度，<br>
        :return: <br>
        """<br>
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25)) <br>
        val_low = data_ser.quantile(0.25) - iqr<br>
        val_up = data_ser.quantile(0.75) + iqr<br>
        rule_low = (data_ser < val_low) <br>
        rule_up = (data_ser > val_up) <br>
        return (rule_low, rule_up), (val_low, val_up) <br>
    data_n = data.copy()<br>
    data_series = data_n[col_name] <br>
    rule, value = box_plot_outliers(data_series, box_scale=scale) <br>
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]] <br>
    print("Delete number is: {}".format(len(index))) <br>
    data_n = data_n.drop(index) <br>
    data_n.reset_index(drop=True, inplace=True) <br>
    print("Now column number is: {}".format(data_n.shape[0])) <br>
    index_low = np.arange(data_series.shape[0])[rule[0]] <br>
    outliers = data_series.iloc[index_low] <br>
    print("Description of data less than the lower bound is:") <br>
    print(pd.Series(outliers).describe())<br>
    index_up = np.arange(data_series.shape[0])[rule[1]] <br>
    outliers = data_series.iloc[index_up] <br>
    print("Description of data larger than the upper bound is:") <br>
    print(pd.Series(outliers).describe())<br>
    fig, ax = plt.subplots(1, 2, figsize=(15, 7)) <br>
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0]) <br>
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1]) <br>
    return data_n<br>
outliers_proc(train_data, 'power', scale=3) <br>
（3）探索性数据分析<br>
/数字特征：power , kilometer , v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14<br>
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ] <br>
/类型特征：name , model , brand , bodyType , fuelType , gearbox , notRepairDamage , regionCode<br>
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage','regionCode',] <br>
numeric_features.append('price') <br>
price_numeric = train_data[numeric_features] <br>
correlation = price_numeric.corr()<br>
f , ax = plt.subplots(figsize = (7, 7)) <br>
plt.title('Correlation of Numeric Features with Price',y=1,size=16) <br>
sns.heatmap(correlation,square = True, vmax=0.8) <br>
（4）特征工程<br>
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce')-pd.to_datetime(data['regDate'],format='%Y%m%d',errors='coerce')).dt.days<br>
data['city'] = data['regionCode'].apply(lambda x : str(x)[:-3]) <br>
train_gb = train_data.groupby("brand")<br>
all_info = {}<br>
for kind, kind_data in train_gb: <br>
    info = {}<br>
    kind_data = kind_data[kind_data['price'] > 0] <br>
    info['brand_amount'] = len(kind_data) <br>
    info['brand_price_max'] = kind_data.price.max()<br>
    info['brand_price_median'] = kind_data.price.median()<br>
    info['brand_price_min'] = kind_data.price.min()<br>
    info['brand_price_sum'] = kind_data.price.sum()<br>
    info['brand_price_std'] = kind_data.price.std()<br>
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2) <br>
    all_info[kind] = info<br>
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})<br>
data = data.merge(brand_fe, how='left', on='brand') <br>
（5）模型
train_y_ln = np.log(train_y + 1) <br>
model = model.fit(train_X, train_y_ln) <br>
model = XGBRegressor(n_estimators = 100, objective='reg:squarederror') <br>
model = model.fit(train_X, train_y) <br>

