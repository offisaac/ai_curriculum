from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import re
import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import MinMaxScaler
import time
from rich.progress import track
#预测步骤
#1.使用df读取清洗后的文件 转化为张量 2.创建训练模型类 3.进行预测
#define
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features=["week_X-2","week_X-3","week_X-4","MA_X-4","dayOfWeek","weekend","holiday","Holiday_ID","hourOfDay","T2M_toc"]
target=["DEMAND"]
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

#获取excel 转化成张量
df=pd.read_excel("datas/short_term_load/train_dataframes.xlsx",engine='openpyxl')
X_train=feature_scaler.fit_transform(df[features])
Y_train=target_scaler.fit_transform(df[target])
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)

df=pd.read_excel("datas/short_term_load/test_dataframes.xlsx",engine='openpyxl')
X_test=feature_scaler.fit_transform(df[features])
Y_test=target_scaler.fit_transform(df[target])
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

dataset_train = TensorDataset(X_train, Y_train)
dataset_test = TensorDataset(X_test, Y_test)
dataloader_train = DataLoader(dataset_train, batch_size=30, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=30, shuffle=True)
# print(next(iter(dataloader_test))[0].shape)#这里使用iter返回迭代对象 使用next调取 使用[0]得到train的第一个batch 返回[100,10]代表sample_num和feature_num 而不包含batch_num
#创建训练模型类别
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义多层LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 前向传播LSTM
        x = x.unsqueeze(1)  # 添加序列长度维度，形状变为 (batch_size, seq_len=1, input_size)
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        # 取最后一个时间步的输出
        y = self.fc(lstm_out[:, -1, :])  # 正确索引最后一个时间步的输出
        return y#这里y的维度是sample_num 1(计算loss那一步应该会对此处理)

# 初始化模型、损失函数和优化器
input_size = len(features)  # 输入特征数量
num_epochs=51
hidden_size = 64  # LSTM的隐藏层大小
output_size = 1  # 输出层的大小（例如预测一个数值）
num_layers = 2  # LSTM的层数
loss_list=[]
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=0)  # 如果没有二阶beta参数 Adam和RMSprop可以认为同效果 学习率0.0001就ok了 这里故意拖慢的
# 训练循环
for epoch in track(range(num_epochs)):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in dataloader_train:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_list.append([epoch,loss.item()])
    avg_loss = epoch_loss / len(dataloader_train)
    if epoch != 0 and (epoch) % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs-1}], Loss: {avg_loss:.4f}')
        np.save('./save_data/save_list/save_list_short_term.npy', np.array(loss_list))
        torch.save(model.state_dict(), f'save_model_parameter/load_pred/model_parameter_{epoch}')
# 评估模型
model.eval()
model.load_state_dict(torch.load(f'save_model_parameter/load_prd/model_parameter_50',weights_only=True))
with torch.no_grad():
    predictions = []
    actuals = []
    for batch_x, batch_y in dataloader_test:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_x)
        predictions.append(output.cpu().numpy())
        actuals.append(batch_y.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
# 反归一化预测值和实际值
predictions = target_scaler.inverse_transform(predictions)
actuals = target_scaler.inverse_transform(actuals)

loss_list=np.load('./save_data/save_list/save_list_short_term.npy')
# 绘制结果
plt.figure(figsize=(12,6))
plt.plot(loss_list[:,0],loss_list[:,1],label="epoch_loss")
plt.legend()
plt.savefig("photos/load_pred/loss_epoch.png")#保存必须给定目标文件类型才停止
plt.show()
plt.figure(figsize=(12, 6))  # 创建第二个图 相当于清除原图层了
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted', color='Pre')
plt.legend()
plt.savefig("photos/load_pred/predict_target.png")#保存必须给定目标文件类型才停止
plt.show()





