import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io
import optuna


# 自定义数据集
class PolynomialDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
class PolynomialModel(nn.Module):
    def __init__(self):
        super(PolynomialModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(64, 64)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(64, 1)   # 第二个隐藏层到输出层
        # 激活函数
        self.relu = nn.ReLU()
        # Initialize weights using Xavier initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))   # 使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)
        return x


# 数据标准化函数
def normalize_data(X, y):
    X_mean = np.mean(X)
    X_std = np.std(X)
    y_mean = np.mean(y)
    y_std = np.std(y)

    X_normalized = (X - X_mean) / X_std
    y_normalized = (y - y_mean) / y_std
    return X_normalized, y_normalized, X_mean, X_std, y_mean, y_std

def train(device, model, dataset_train, dataset_test, epochs, lr):
    criterion = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
    train_losses = []
    test_losses = []
    model.to(device)
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        epoch_train_loss = 0.0  # 训练集损失

        for X_batch, y_batch in train_loader:
            # 前向传播
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 反向传播
            optimizer.step()       # 更新参数

            epoch_train_loss += loss.item() * X_batch.size(0)  # 累加损失
        epoch_train_loss /= len(dataset_train)  # 计算平均损失
        train_losses.append(epoch_train_loss)

        # 测试集评估
        model.eval()  # 设置模型为评估模式
        epoch_test_loss = 0.0  # 测试集损失

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_test_loss += loss.item() * X_batch.size(0)  # 累加损失

        epoch_test_loss /= len(dataset_test)  # 计算平均损失
        test_losses.append(epoch_test_loss)

        # 打印损失信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}')

    # 绘制训练集和测试集的损失变化
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Train and Test Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == '__main__':
    N = 3  # <--倍数改这里
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_values = scipy.io.loadmat('train9.mat')
    y_values = y_values['train9']
    # non_zero_indices = np.where(y_values != 0)[0][0]-1  # train1~8.mat
    non_zero_indices = np.where(y_values != 0)[0][0]  # train9.mat(开头没有0值）
    x_values = np.arange(0,len(y_values)-non_zero_indices)/10
    y_values = y_values[non_zero_indices:]

    sampling_rate = 0.1*N
    indices = np.arange(0, len(y_values), int(sampling_rate / 0.1))
    train_indices = indices
    test_indices = np.setdiff1d(np.arange(len(y_values)), indices)

    X_train= x_values[train_indices].reshape(-1, 1)
    y_train = y_values[train_indices].reshape(-1, 1)
    # 数据标准化
    X_normalized_train, y_normalized_train, X_mean, X_std, y_mean, y_std = normalize_data(X_train, y_train)

    X_test = x_values[test_indices].reshape(-1, 1)
    y_test = y_values[test_indices].reshape(-1, 1)
    X_normalized_test, y_normalized_test, _, _, _, _ = normalize_data(X_test, y_test)

    # 转换为数据集
    train_dataset = PolynomialDataset(X_normalized_train, y_normalized_train)
    test_dataset = PolynomialDataset(X_normalized_test, y_normalized_test)

    # 创建模型
    model = PolynomialModel()

    # 训练模型
    num_epochs = 150  # <---Epoch改这里
    learning_rate = 0.001

    train(device, model, train_dataset, test_dataset, num_epochs, learning_rate)
    model.eval()  # 设置模型为评估模式
    X_test_tensor = torch.FloatTensor(X_normalized_test)
    y_test_tensor = torch.FloatTensor(y_test)
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = y_pred * y_std + y_mean
    # 可视化预测结果
    plt.figure(figsize=(10, 5))
    plt.plot(X_test_tensor.numpy(), y_test_tensor.numpy(), label='True function', color='blue')
    plt.scatter(X_test_tensor.numpy(), y_pred.numpy(), color='red', label='Predicted values', s=0.5)
    plt.legend()
    plt.title('Function Approximation using Neural Network')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()
    y_total = np.zeros(1200)
    y_train = y_train.tolist()
    y_pred = y_pred.tolist()
    for i in range(1200-non_zero_indices):
        if i % N == 0:
            y_total[i+non_zero_indices] = y_train.pop(0)[0]
        else:
            y_total[i+non_zero_indices] = y_pred.pop(0)[0]

    # 检查还原效果
    x_values = np.arange(0, 120, 0.1)
    plt.scatter(x_values, y_total, color='red', label='Predicted values', s=0.5)
    y_real = scipy.io.loadmat('train9.mat')
    y_real = y_real['train9']
    plt.scatter(x_values, y_real, label='True function', color='blue',s=0.5)
    plt.title('Function Approximation using Neural Network')
    plt.legend()
    plt.show()
    # 将y_total存成mat文件
    scipy.io.savemat('y_total9.mat', {'y_total': y_total})






