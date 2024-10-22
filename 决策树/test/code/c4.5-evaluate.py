import pandas as pd
from sklearn.model_selection import train_test_split
from C45 import C45Classifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io as sio


def evaluate(datas,labels):
    data_tr, data_te, label_tr, label_te = train_test_split(datas, labels, test_size=0.1)
    # 构造C4.5决策树
    model = C45Classifier()
    model.fit(data_tr, label_tr)
    pre_te = model.predict(data_te)
    return accuracy_score(label_te,pre_te)


if __name__ == '__main__':
    df = pd.read_csv('../../extent/environment_data.csv')
    last_column = df.iloc[:, -1]
    unique_values = last_column.unique()
    value_to_number = {value: idx + 1 for idx, value in enumerate(unique_values)}
    df.iloc[:, -1] = last_column.map(value_to_number)
    array = df.values.tolist()
    labels = [row[-1] for row in array]
    datas = [row[:-1] for row in array]
    results = []
    for i in tqdm(range(1000),desc="多次训练中"):
        results.append(evaluate(datas,labels))
    # 绘制直方图
    plt.hist(results, bins=30, edgecolor='black')  # bins 表示分成30个区间
    plt.xlabel('Value')  # x轴标签
    plt.ylabel('Frequency')  # y轴标签
    plt.title('Histogram of Observed Values')  # 图表标题
    plt.show()
    sio.savemat('Accuraries.mat', {'observed_values': results})