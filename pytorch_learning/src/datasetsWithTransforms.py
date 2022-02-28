import torchvision

# 打包一个操作动作合集
# 这里使用转换tensor类型操作
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='../../../dataset/CIFAR10',
                                         transform=dataset_transform,
                                         train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='../../../dataset/CIFAR10',
                                        transform=dataset_transform,
                                        train=False, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
#
# print(test_set[0]) # RGB类型 这里的tensor类型还有个标签参数

# 实例化
writer = SummaryWriter('logs')
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()
