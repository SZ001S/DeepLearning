import torch
from torch.nn import MaxPool2d
from torch.nn import Conv2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = torch.nn.Sequential(
            Conv2d(3, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, (5, 5), padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = Model()
print(model)
inputs = torch.randn(size=(64, 3, 32, 32))
output = model(inputs)
print(output.shape)

writer = SummaryWriter('logs')
writer.add_graph(model, inputs)
writer.close()
