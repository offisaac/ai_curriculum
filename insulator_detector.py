import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time

from rich.progress import track
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def custom_Cross_Entropy(logits, target):
#     softmax_output = F.softmax(logits,
#                                dim=1)  # dim=1代表有两个维度 这里单个logit就是一个维度了(len=10) logits又是另外一个维度 同时 这里也介绍了F的一个用处 其内定义好很多基础函数
#
#     log_softmax_output = -torch.log(softmax_output)  # 转化为log形式数据 返回张量对象
#     batch_loss = log_softmax_output.gather(1, target.unsqueeze(1)).squeeze(
#         1)  # squeeze用于修改向量维度unsqueeze表示产生序列1维度 squeeze表示消去序列1维度 底层操作其实就是将行向量变成列向量 gather第一个参数表示对数据的列进行操作 本身代表通过给定给定数据序列 来选取每一列中的第几个序列元素
#     loss = batch_loss.mean()  # 对列向量求均值
#     return loss


compose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])
to_tensor = transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=compose_transform)
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=compose_transform)



# class mydataset(dataset.Dataset):
#     def __init__(self, dataset):
#         # 初始化变量
#         self.idx_to_class = {}
#         self.Y_unique_heating_code = []
#
#         # 构建类索引映射
#         for key in dataset.class_to_idx:
#             self.idx_to_class[dataset.class_to_idx[key]] = key
#
#         # 产生独热编码
#         unique_heating_code = torch.eye(10)  # 独热编码矩阵
#         for i in dataset.targets:
#             self.Y_unique_heating_code.append(unique_heating_code[i])  # 创建真实值对应的独热编码
#         self.Y_unique_heating_code = torch.stack(self.Y_unique_heating_code)  # 转换为张量
#
#         # 获取数据
#         self.X = dataset.data
#         self.class_num = len(dataset.class_to_idx)
#
#     def __getitem__(self, index):
#         # 获取输入图像数据及其独热编码标签
#         return torch.tensor(self.X[index], dtype=torch.float32).permute(2, 0, 1), self.Y_unique_heating_code[index]
#
#     def __len__(self):
#         return len(self.X)
class mydataset(dataset.Dataset):
    def __init__(self, dataset):
        self.idx_to_class = {}
        self.Y_unique_heating_code = []
        for key in dataset.class_to_idx:
            self.idx_to_class[dataset.class_to_idx[key]] = key  # 供最后一步独热编码取最大值对应使用
            unique_heating_code = torch.tensor(np.eye(10))  # 产生独热编码
            for i in dataset.targets:
                self.Y_unique_heating_code.append(unique_heating_code[:, i])  # 创建真实值对应独热编码
            self.X = dataset.data  # data本身就是双array 对应元素是三通道 卷积层可以直接读取 这部分底层不需要人为操作
            self.Y = dataset.targets
            self.class_num = len(dataset.class_to_idx)

    def __getitem__(self, index):
        # return torch.tensor(self.X[index],dtype=torch.float32).permute(2,0,1),self.Y_unique_heating_code[index]
        return torch.tensor(self.X[index], dtype=torch.float32).permute(2, 0, 1), self.Y[
            index]  # 理论上应该使用独热编码 但是pytorch对数学公式做了处理 先得到输出的权重 再softmax得到概率 再取-log从取最大值到取最小值 再根据整数序列选取使用哪个输出概率(如果模型好 真实对应的就是最大的概率就是最小的log(P)输出

    def __len__(self):
        return len(self.X)


my_train_data = mydataset(train_data)
my_test_data = mydataset(test_data)
my_train_data_loader = torch.utils.data.DataLoader(my_train_data, batch_size=400, shuffle=True)
my_test_data_loader = torch.utils.data.DataLoader(my_test_data, batch_size=400, shuffle=True)


class CNNClassifier(nn.Module):
    def __init__(self, class_num):
        super().__init__()  # !外部的输入不应该放在类名后(作为继承) 而是放在init后
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)  # 这里padding=1是指每个边都向外扩一个"0"的像素 所以对于行 相当于扩了两个像素
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1,
                               padding=1)  # 实际设计图片的卷积层时 通道输出图层数由filter决定 最后给线性层输入的列向量行数字需要根据卷积后的图片尺寸决定 这个有公式计算
        self.conv3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # 卷积操作影响图层个数和图的尺寸 池化只影响图的尺寸 图最终尺寸只和初始图尺寸 卷积的stride padding有关 和通道无关 通道只和图层有关
        self.func1 = nn.Linear(64 * 2 * 2, 128)  # 怎么从32通道8*8array对应到32*8*8个列向量 底层不用考虑
        self.func2 = nn.Linear(128, 128)
        self.func3 = nn.Linear(128, class_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # Dropout用于训练更新参数时随机的让神经元的输出置0 但实际使用和评估时不做改变 因为CNN参数相对于全连接不太多 所以一般给的这个值不大
        self.sigmoid = nn.Sigmoid()

    def forward(self, Input):
        x = self.conv1(Input)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.dropout(x)  # 主要是为了防止过拟合使用 防止模型对某些神经元过于依赖 增加泛化能力(希望部分集训练的参数对全集也适应)
        x = x.view(x.size(0),
                   -1)  # x = x.view(-1,32*8*8) 这里-1可以在任意位置 表示该位置由输入和另一个位置求得 这里两式等效 将 x 的形状从 [batch size, 32, 8, 8] 变为 [batch size, 2048]。
        x = self.func1(x)
        x = self.func2(x)
        x = self.func3(x)
        # x = self.relu(x) #不用加 使用crossentropy 计算Loss内部会对Outputs先进行softmax再计算
        return x


model = CNNClassifier(my_train_data.class_num).to(device)
for param in model.parameters():
    if param.dim() > 1:  # 只对多维参数（如权重矩阵）应用 Xavier 初始化
        torch.nn.init.xavier_uniform_(param)

model.load_state_dict(
    torch.load(f'./model_parameter_set/model_parameter_50',
               weights_only=True))

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0)
# optimizer = optim.RMSprop(model.parameters(), lr=0.0005, momentum=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=0)  # 如果没有二阶beta参数 Adam和RMSprop可以认为同效果
epochs = 500
loss_value = 0
for epoch in track(range(epochs), description=f"[red]loss {loss_value}"):
    for X, Y in my_train_data_loader:
        X, Y = X.to(device), Y.to(device)  # 将数据移动到设备上
        outputs = model(X)
        # loss = criterion(outputs, Y).to(device)
        loss = custom_Cross_Entropy(outputs, Y).to(device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    loss_value = loss.item()
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    if (epoch) % 10 == 0:
        torch.save(model.state_dict(), f'model_parameter_set/model_parameter_{epoch}')
    if loss.item() < 0.001:
        break

# loaded_model = CNNClassifier(class_num=10).to(device)
# loaded_model.load_state_dict(
#     torch.load(f'./model_parameter_set/model_parameter_420',
#                weights_only=True))
# loaded_model.eval()  # 设置为评估模式 用于关闭某些正则化操作

# print(next(loaded_model.parameters()).device)#这里是使用next读取可迭代对象第一个参数所在的位置 然后对该参数判断在cpu还是在gpu上


correct = 0
total = 0
with torch.no_grad():  # 禁用梯度计算，以减少内存占用
    for data, labels in my_train_data_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)  # 获取每个样本的最大概率的索引，即预测的类别 1代表从行取 也就是batch计算出的结果其实是行堆叠
        total += labels.size(0)  # 表示在张量第零维的长度 张量本身的shape返回其各个维度的长度！！！
        correct += (
                predicted == labels).sum().item()  # ==是两个vector判断等 相等处赋True 不等处赋值False 返回相同大小vector .sum是将其内所有数相加 .item是将张量类型数转化成普通类型数
print(f"the possibility of rightness is{correct / total}")