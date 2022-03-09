#  import Package
import torch
# -------------------------------
# 会用到relu函数
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
# 与数据集相关 DataLoader相关
# ------------------------------
from torchvision import transforms

# 1.Prepare Dataset
batch_size = 64
# 转换器 将PIL Image 像素值在[0, 255]
# 转换为PyTorch Tensor 像素值在[0, 1]
# 图片是WXHXC形式 Tensor里面则是CXWXH形式
# Normalize是标准化 第一个为mean 第二个为std
# 0.1307和0.3081是根据MNIST数据集的分布计算得来的
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    # 需要实现这个forward函数 这是构建计算图 
    # 不同于构建模型 这里需要告知神经网络需要
    # 进行什么计算
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


# 实例化模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

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
            print(f'[{epoch + 1}, {batch_index + 1} loss: {running_loss / 300:.3f}]')
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
    print('Accuracy on test set: %d %%'
          % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        # 一轮下来 训练测试同时进行
        train(epoch)
        # 也可以 if epoch % 10 == 9: 
        # 表示经过10轮测试一次
        test()
