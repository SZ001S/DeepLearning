import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('../../../dataset/CIFAR10',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3),
                                     stride=(1, 1), padding=0)


    def forward(self, x):
        x = self.conv1(x)
        return x

model = Model()

writer = SummaryWriter('logs')

step = 0
for data in dataloader:
    imgs, targets = data
    outputs = model(imgs)
    print(imgs.shape)
    print(outputs.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images('inputs', imgs, step)
    # torch.Size([64, 6, 30, 30]) -> [-1, 3, 30, 30]

    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images('outputs', outputs, step)
    step += 1

writer.close()
