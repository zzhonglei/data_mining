#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

inputfile = '../data/data.csv'  # 输入的数据文件
data = pd.read_csv(inputfile)  # 读取数据
lasso = Lasso(alpha=1000,max_iter=10000)  # 调用Lasso()函数，设置λ的值为1000
lasso.fit(data.iloc[:,0:13],data['y'])
print('相关系数为：',np.round(lasso.coef_,5))  # 输出结果，保留五位小数

print('相关系数非零个数为：',np.sum(lasso.coef_ != 0))  # 计算相关系数非零的个数

mask = lasso.coef_ != 0  # 返回一个相关系数是否为零的布尔数组
print('相关系数是否为零：',mask)

outputfile ='../tmp/new_reg_data.csv'  # 输出的数据文件
# 获取前 13 列非零相关系数的数据
new_reg_data = data.iloc[:, 0:13].iloc[:, mask]  # 只对前13列特征数据进行筛选
# 然后可以选择将目标变量 'y' 列添加回来
new_reg_data['y'] = data['y']  # 添加目标列
# 存储筛选后的新数据
new_reg_data.to_csv(outputfile)
print('输出数据的维度为：',new_reg_data.shape)  # 查看输出数据的维度
