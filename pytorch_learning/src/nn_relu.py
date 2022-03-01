import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

inputs = torch.tensor([[1, -0.5],
                       [-1, 3]])
inputs = torch.reshape(inputs, (-1, 1, 2, 2))
print(inputs.shape)

dataset = torchvision.datasets.CIFAR10('../../../dataset/CIFAR10', train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = torch.nn.ReLU()  # inplace参数表示是否进行原地替换 默认为False可以保留数据
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(x)
        return x


model = Model()
# print(model(inputs))

writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('inputs', imgs, step)
    outputs = model(imgs)
    writer.add_images('outputs', outputs, step)
    step += 1

writer.close()
