import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt

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

    x_pre = data[feature].loc['2014':].copy()
    x_pre = (x_pre - data_mean) / data_std
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_pre = torch.tensor(x_pre[feature].values, dtype=torch.float32)
    return x_train, y_train, x_pre,data_mean,data_std

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # 使用 Xavier 均匀分布初始化权重
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 将偏置初始化为 0

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=8, out_features=64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(in_features=64, out_features=32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
            # nn.Linear(in_features=8, out_features=32), nn.BatchNorm1d(32),nn.ReLU(),
            # nn.Linear(in_features=32, out_features=128), nn.BatchNorm1d(128),nn.ReLU(),
            # nn.Linear(in_features=128, out_features=512), nn.BatchNorm1d(512),nn.ReLU(),
            # nn.Linear(in_features=512, out_features=128), nn.BatchNorm1d(128),nn.ReLU(),
            # nn.Linear(in_features=128, out_features=32), nn.BatchNorm1d(32),nn.Sigmoid(),
            # nn.Linear(in_features=32, out_features=1)
        )
    def forward(self, X):
        return self.net(X)
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            pre = self.net(X)
        return pre

def train(net,x_train,y_train,loss,num_epoch,trainer,scheduler,device):
    Loss = []
    for epoch in range(num_epoch):
        if isinstance(net, torch.nn.Module):
            net.train()
        if device is not None:
            net.to(device)
            x_train = x_train.to(device)
            y_train = y_train.to(device)
        y_hat = net(x_train)
        l = loss(y_hat, y_train)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {l.item():.12f}')
        scheduler.step(l)
        Loss.append(l.item())
    return Loss

if __name__ == '__main__':
    if __name__ == '__main__':
        inputfile = '../tmp/new_reg_data_GM11.xlsx'
        x_train, y_train, x_pre, data_mean, data_std = readData(inputfile)
        # 创建模型
        model = MLP()
        model.apply(init_weight)  # 初始化模型权重
        # 定义损失函数和优化器
        loss = nn.MSELoss()  # 均方误差损失
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Adam 优化器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,threshold=1e-3)

        # 设置训练的参数
        epochs = 1000
        batch_size = 32
        loss_values = train(model,x_train,y_train,loss,epochs,optimizer,scheduler,torch.device('mps'))

        model.to(torch.device('cpu'))
        predictions = model.predict(x_pre)  # 使用封装的预测方法
        predictions_original = predictions * data_std['y'] + data_mean['y']
        original_data = y_train*data_std['y'] + data_mean['y']
        print(f'Predictions (original scale): {original_data.tolist()}')
        print(f'Predictions (original scale): {predictions_original.tolist()}')
        print("\n\n")

        predictions_original = model.predict(x_train) * data_std['y'] + data_mean['y']
        for len in range(len(original_data.tolist())):
            print(original_data.tolist()[len],'->',predictions_original.tolist()[len])

        # 绘制折线图并使用对数坐标
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Training Loss', color=(0.2,0.6,0.8),linewidth=2)
        plt.title('Training Loss over Epochs',fontsize=24)  # 图表标题
        plt.xlabel('Epochs',fontsize=12)  # X轴标签
        plt.ylabel('Loss (log scale)',fontsize=12)  # Y轴标签
        plt.yscale('log')  # 设置Y轴为对数刻度
        plt.legend(fontsize=18)  # 显示图例
        # 设置边框的宽度
        ax = plt.gca()  # 获取当前轴
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # 设置边框宽度为2
        # 展示图像
        plt.show()
        # 创建一个 DataFrame 将损失值存储为一列
        df = pd.DataFrame(loss_values, columns=['Loss'])
        # 将 DataFrame 存储到 CSV 文件中
        df.to_csv('loss_values.csv', index=False)
        torch.save(model.state_dict(), 'params.pt')