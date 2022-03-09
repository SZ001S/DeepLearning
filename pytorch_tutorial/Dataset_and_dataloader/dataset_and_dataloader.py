import torch
from torch.utils.data import DataLoader
# 这里的Dataset是抽象类 不能实列化只能被子类继承
from torch.utils.data import Dataset
import numpy as np


# 继承父类Dataset一些基本功能 并实现相应的一些方法
class DiabetesDataset(Dataset):
    def __init__(self, filepath) -> None:
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # 魔法方法magic function
    # 用来支持一些下标操作
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # This magic function returns length of dataset
    def __len__(self):
        return self.len


# dataset = DiabetesDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=32,
#                           shuffle=True,
#                           num_workers=2)

dataset = DiabetesDataset('pytorch_tutorial\课件\PyTorch深度学习实践\diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32,
                          shuffle=True, num_workers=2)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
# -------------------------------------------------#

criterion = torch.nn.BCELoss(reduction='mean')

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# -------------------------------------------------#


for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1. Prepare data
        inputs, labels = data
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()
