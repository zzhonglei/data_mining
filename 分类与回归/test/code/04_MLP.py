import pandas as pd

def readData(inputfile):
    inputfile = '../tmp/new_reg_data_GM11.xlsx'  # 灰色预测后保存的路径
    data = pd.read_excel(inputfile)  # 读取数据
    feature = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']  # 属性所在列
    # 给第一列命名为 'year'
    data.columns = ['year'] + list(data.columns[1:])  # 给第一列命名为 'year'
    # 将第一列（年份列）设置为索引
    data.set_index('year', inplace=True)
    # 选择1994到2013年的数据
    data_train = data.loc[range(1994, 2014)].copy()
    data_mean = data_train.mean()
    data_std = data_train.std()
    data_train = (data_train - data_mean) / data_std  # 数据标准化
    x_train = data_train[feature].values  # 属性数据
    y_train = data_train['y'].values  # 标签数据

    data_pre = data.loc['2014':].copy()
    data_mean = data_pre.mean()
    data_std = data_pre.std()
    data_pre = (data_pre - data_mean) / data_std  # 数据标准化
    x_pre = data_pre[feature].values  # 属性数据



if __name__ == '__main__':
    inputfile = '../tmp/new_reg_data_GM11.xlsx'
    readData(inputfile)