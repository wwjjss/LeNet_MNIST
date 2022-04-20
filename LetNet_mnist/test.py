import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np


class LeNet(nn.Module):
    def __init__(self, num_class=10):
        super(LeNet, self).__init__()  # 继承父类所有属性和方法，父类属性用父类的方法初始化
        self.conv1 = nn.Sequential(  # nn.Sequential:相当于一个容器，将一系列操作包含其中
            nn.Conv2d(1, 6, 5, 1, 2),  # Conv2d(in_channels,out_channels,kernel_size,stride,padding)  out_size(6*28*28)
            nn.ReLU(),  # ReLu()激活函数引入非线性，把负值变为0，正值不变
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，MaxPool2d(kernel_size,stride,padding)  out_size(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),  # out_size(16*10*10)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # out_size(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 全连接层nn.Linear(in_features,out_features)输入和输出的二维张量大小或神经元个数
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)  # 最后一层得到要数字分类的10类概率值

    # 前向传播

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)  # 表示将多维的tensor数据展平成一维，torch.view(a,b):重构成a*b维的张量 torch.view(a,-1):-1表示列需要自动计算列数
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 读取图片
test_img = 'newTest6.png'
img = plt.imread(test_img)
images = Image.open(test_img)
images = images.resize((28, 28))
images = images.convert('L')
transform = transforms.ToTensor()
images = transform(images)
images = images.resize(1, 1, 28, 28)

# 加载网络模型
model = LeNet()
model.load_state_dict(torch.load('LeNet.pth'))
# model = torch.load('LeNet.pth')
model.eval()
outputs = model(images)

values, indices = outputs.data.max(1)  # 返回最大概率值和下标
plt.title('{}'.format((int(indices[0]))))
plt.imshow(img)
plt.show()
