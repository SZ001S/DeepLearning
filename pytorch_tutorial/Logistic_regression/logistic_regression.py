import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
#----------------------------------------------------------#


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self) -> None:
        # sigmoid函数是没有参数可训练的
        # 拿来即用就行了
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()


#---------------------------------------------------------#
# 二分类的交叉熵损失
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#---------------------------------------------------------#


for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#------------------------------------------------------------------#
# 导入数据集的操作
# import torchvision
# 导入数据集操作
# train_set = torchvision.datasets.MNIST(root='../datasets/mnist', \
#     train=True, download=True)
# test_set = torchvision.datasets.MNIST(root='../datasets/mnist', \
#     train=False, download=True)
#------------------------------------------------------------------#


# 每周学习10小时 采样200个点
x = np.linspace(0, 10, 200)
# 将样本变为200行1列的矩阵 view()类似numpy的reshape()函数
x_t = torch.Tensor(x).view((200, 1))
# 将得到的张量送入model中
y_t = model(x_t)
# 拿出y_t数据可以直接调用numpy()得到ndarray数组
y = y_t.data.numpy()
# 拿到数组即可绘图
# 蓝色线
plt.plot(x, y)
# 红色线
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()