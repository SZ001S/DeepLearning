import torchvision

# 准备的测试集合
test_data = torchvision.datasets.CIFAR10('../../../dataset/CIFAR10', train=False,
                                         transform=torchvision.transforms.ToTensor())

