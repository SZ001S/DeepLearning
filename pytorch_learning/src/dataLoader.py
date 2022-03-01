import torchvision

# 准备的测试集合
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('../../../dataset/CIFAR10', train=False,
                                         transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

# 测试集中的第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)


writer = SummaryWriter('logs')
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        '''Add batched image data to summary'''
        writer.add_images('Epoch: {}'.format(epoch), imgs, step)
        step += 1

writer.close()
