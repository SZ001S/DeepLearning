import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
#-------------------------------
# 会用到relu函数
import torch.nn.functional as F
import torch.optim as optim
# in_channels, out_channels = 5, 10
# width, height = 100, 100
# kernel_size = 3
# batch_size = 1

# # 一个4维Tensor
# # 数值是服从标准正态分布的 从中采样的
# input = torch.randn(
#     batch_size,
#     in_channels,
#     width,
#     height
# )

# # 必要的三个参数：
# # 输入通道和输出通道以及卷积核的形状 
# # 最重要的是通道数要对齐 否则会出错
# conv_layer = torch.nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size=kernel_size
# )

# output = conv_layer(input)

# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)


# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1]

# # BXCXWXH view四个值顺序代表含义 
# # batch_size channel width height
# input = torch.Tensor(input).view(1, 1, 5, 5)

# # conv_layer = torch.nn.Conv2d(
# #     1, 1, kernel_size=3, padding=1, bias=False, 
# # )

# conv_layer = torch.nn.Conv2d(
#     1, 1, kernel_size=3, bias=False, 
#     stride=2
# )


# # 这里的view四个值分别表示 
# # 输出channels 输入channels width heigh
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)

# conv_layer.weight.data = kernel.data

# output = conv_layer(input)
# print(output)


# input = [3,4,6,5,
#          2,4,6,8,
#          1,6,7,8,
#          9,7,4,6]

# input = torch.Tensor(input).view(1,1,4,4)

# # subsampling的一种：maxpooling
# # kernel_size=2的时候 stride也会默认设置为2
# maxpooling_layer = torch.nn.MaxPool2d(kernel_size=(2,2))

# output = maxpooling_layer(input)

# print(output)

# 1.Prepare Dataset
batch_size = 64
# 转换器 将PIL Image 像素值在[0, 255]
# 转换为PyTorch Tensor 像素值在[0, 1]
# 图片是WXHXC形式 Tensor里面则是CXWXH形式
# Normalize是标准化 第一个为mean 第二个为std
# 0.1307和0.3081是根据MNIST数据集的分布计算得来的
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(
    root='../dataset/mnist/', 
    train=True, 
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    shuffle=True, 
    batch_size=batch_size
)

test_dataset = datasets.MNIST(
    root='../dataset/mnist/', 
    train=False, 
    download=True,
    transform=transform
)

# 这里的shuffle=False保证了实验的观察环境条件一致
test_loader = DataLoader(
    train_dataset, 
    shuffle=False, 
    batch_size=batch_size
)

# 2.Design Model
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            1, 10, kernel_size=(5,5)
        )
        self.conv2 = torch.nn.Conv2d(
            10, 20, kernel_size=(5,5)
        )
        self.pooling = torch.nn.MaxPool2d(kernel_size=(2,2))
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) 
        # to (n, 784)
        # x是Tenso类型 先取第一个维度的值 
        # 也就是样本数量
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        # 矫正一下x的形状放入全连接层
        x = x.view(batch_size, -1) # flatten
        # 交叉熵损失集成了softmax激活 不用激活
        x = self.fc(x)
        return x

# model = Net().to(
#     device=torch.device(
#         'cuda:0' if torch.cuda.is_available() else 'cpu'
#     )
# )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Net()
model.to(device)


# 3.Construct Loss and Optimizer
# 在使用这些工具时又会在构建计算图
# 实例化一个Loss器
criterion = torch.nn.CrossEntropyLoss().to(device='cuda:0')
# 实列化一个优化器
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.5
)


# 4.Train and Test
# 把一轮循环封装到函数中
def train(epoch):
    running_loss = 0.0
    for batch_index, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 300 == 299:
            # running_loss累计300个样本训练后取平均输出
            print(f'[{epoch+1}, {batch_index+1} loss: {running_loss / 300:.3f}]')
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    # 防止构建计算图 test过程不是train过程 
    # 不是构建计算图求梯度优化模型的过程
    # 没有反向传播步骤 只需算一下forward过程
    with torch.no_grad():
        # 使用第119行代码后 
        # 里面的代码无需计算梯度
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # 输出是一个一行行记录的矩阵 
            # 找每一行的最大值输出来
            # 这是计算cross entropy的结果
            outputs = model(images)
            # 第一个images数据没有用
            # 就用_ 这个没有名字的变量代替
            # 沿着dim=1的维度（按行）找最大值的下标 
            # 返回值有两个；一个是值，一个是下标
            _, predicted = torch.max(outputs.data, dim=1)
            # .size(0)就是labels这个矩阵的dim=0的分量数值大小
            total += labels.size(0)
            # 比较相等 真为1 假为0
            correct += (predicted == labels).sum().item()   
    print('Accuracy on test set: %d %%' \
        % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        # 一轮下来 训练测试同时进行
        train(epoch)
        # 也可以 if epoch % 10 == 9: 
        # 表示经过10轮测试一次
        test()

###############################################
# pytorch使用显卡非常简单 
# 就是模型和数据往显卡一扔就行了（必须是同一块显卡）
###############################################

