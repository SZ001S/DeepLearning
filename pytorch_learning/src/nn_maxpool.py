import torch

# inputs = torch.tensor([[1, 2, 0, 3, 1],
#                        [0, 1, 2, 3, 1],
#                        [1, 2, 1, 0, 0],
#                        [5, 2, 3, 1, 1],
#                        [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# inputs = torch.reshape(inputs, (-1, 1, 5, 5))
# print(inputs.shape)
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10('../../../dataset/CIFAR10', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(datasets, batch_size=64)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 最大池化的时候是否保留
        # 卷积核超出tensor边缘的问题
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool(x)
        return x

model = Model()

writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('inputs', imgs, step)
    outputs = model(imgs)
    writer.add_images('outputs', outputs, step)
    step += 1
    
writer.close()
# model = Model()
# outputs = model(inputs)
# print(outputs)
