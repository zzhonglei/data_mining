from MLP import MLP
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def readData(inputfile):
    data = pd.read_excel(inputfile)  # 从指定路径读取数据
    feature = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']
    data.columns = ['year'] + list(data.columns[1:])
    data.set_index('year', inplace=True)
    # 选择1994到2013年的数据
    data_train = data.loc[range(1994, 2014)].copy()
    data_mean = data_train.mean()
    data_std = data_train.std()
    data_train = (data_train - data_mean) / data_std  # 数据标准化

    x_train = data_train[feature].values  # 属性数据
    y_train = data_train['y'].values  # 标签数据

    x_pre = data[feature].copy()
    x_pre = (x_pre - data_mean) / data_std
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_pre = torch.tensor(x_pre[feature].values, dtype=torch.float32)
    return x_train, y_train, x_pre,data_mean,data_std

if __name__ == '__main__':
    net = MLP()
    net.load_state_dict(torch.load("params.pt",weights_only=True))
    inputfile = '../tmp/new_reg_data_GM11.xlsx'
    x_train, y_train, x_pre,data_mean,data_std = readData(inputfile)
    predictions = net.predict(x_pre)  # 使用封装的预测方法
    predictions_original = predictions * data_std['y'] + data_mean['y']
    original_data = x_pre * data_std['y'] + data_mean['y']
    original_data = y_train*data_std['y'] + data_mean['y']
    for len in range(len(original_data.tolist())):
        print(original_data.tolist()[len], '->', predictions_original.tolist()[len])
    # 读取 CSV 文件
    df = pd.read_excel('../tmp/new_reg_data_GM11_revenue.xlsx')
    # 获取最后两列数据
    y = df.iloc[:, -2].values[:-2] # 真实值
    s_y = df.iloc[:, -1].values[:-2] # svr预测值
    m_y = predictions_original.reshape(-1).numpy()[:-2] # MLP预测值

    # 计算 SVR 预测值的评价指标
    mse_svr = mean_squared_error(y, s_y)
    rmse_svr = np.sqrt(mse_svr)
    mae_svr = mean_absolute_error(y, s_y)
    r2_svr = r2_score(y, s_y)
    # 计算 MLP 预测值的评价指标
    mse_mlp = mean_squared_error(y, m_y)
    rmse_mlp = np.sqrt(mse_mlp)
    mae_mlp = mean_absolute_error(y, m_y)
    r2_mlp = r2_score(y, m_y)

    # 打印结果
    print("SVR 预测值评价:")
    print(f"均方误差 (MSE): {mse_svr}")
    print(f"均方根误差 (RMSE): {rmse_svr}")
    print(f"平均绝对误差 (MAE): {mae_svr}")
    print(f"R² (决定系数): {r2_svr}")

    print("\nMLP 预测值评价:")
    print(f"均方误差 (MSE): {mse_mlp}")
    print(f"均方根误差 (RMSE): {rmse_mlp}")
    print(f"平均绝对误差 (MAE): {mae_mlp}")
    print(f"R² (决定系数): {r2_mlp}")

    # 创建一个包含真实值、SVR预测值和MLP预测值的数据框
    data = {
        '真实值': y,
        'SVR预测值': s_y,
        'MLP预测值': m_y
    }
    # 将数据转换为DataFrame
    df_output = pd.DataFrame(data)
    # 将 DataFrame 写入 CSV 文件
    df_output.to_csv('../tmp/result.csv', index=False, encoding='utf-8')
