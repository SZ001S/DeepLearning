from turtle import forward
import numpy as np
import torch


xy = np.loadtxt('pytorch_tutorial/Mutiple_dimension_input/diabetes.csv.gz', delimiter=',', dtype=np.float32)
# 最后一列（结果列）不拿
x_data = torch.from_numpy(xy[:, :-1])
# 中括号再嵌套一个中括号保证y_data是一个NX1的矩阵
# 而不是一个向量 感觉问题不大 主要是为了矩阵计算时
# 能对齐
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activat = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activat(self.linear1(x))
        x = self.activat(self.linear2(x))
        x = self.activat(self.linear3(x))
        return x


model = Model()
#-------------------------------------------------#

criterion = torch.nn.BCELoss(reduction='mean')

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#-------------------------------------------------#


for epoch in range(100):
    # Forward
    # 这里没有用到Mini-Batch 用的是一整个Batch
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Updata
    optimizer.step()


