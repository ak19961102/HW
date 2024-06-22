import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 读取CSV文件
data = pd.read_csv('2.csv')

# 提取特征和目标
features = data[['f1', 'f2', 'f3']].values
targets = data['y3'].values

# 定义一个函数来创建时间窗口
def create_sequences(input_data, window_size):
    sequences = []
    for i in range(len(input_data) - window_size):
        sequence = input_data[i:i+window_size]
        sequences.append(sequence)
    return np.array(sequences)

# 定义模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

# 设置超参数
input_size = 3
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001
batch_size = 64
window_size = 10

# 创建训练数据
sequences = create_sequences(features, window_size)
X = torch.tensor(sequences, dtype=torch.float32)
y = torch.tensor(targets[window_size:], dtype=torch.float32).view(-1, 1)

# 划分训练集和测试集
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_dataset = TensorDataset(X[:train_size], y[:train_size])
test_dataset = TensorDataset(X[train_size:], y[train_size:])

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # 测试模型
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for batch_X, batch_y in test_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         outputs = model(batch_X)
#         total += batch_y.size(0)
#         correct += (torch.abs(outputs - batch_y) < 0.5).sum().item()

#     print(f'Test Accuracy: {100 * correct / total}%')
    
# 测试模型并计算RMSE
model.eval()
with torch.no_grad():
    squared_error = 0
    total = 0
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        squared_error += torch.sum((outputs - batch_y) ** 2).item()
        total += batch_y.size(0)

    rmse = np.sqrt(squared_error / total)
    print(f'Test RMSE: {rmse}')







