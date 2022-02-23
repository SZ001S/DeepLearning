import torch


# dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# model
# torch.nn.Model这个父类有好多方法在构建新模型和模型训练时候
# 会用到的 所以要继承这个Module这个模块
class LinearModel(torch.nn.Module):
    # 构造函数
    def __init__(self) -> None:
        # super()调用父类的__inin__()来构造
        # 这一步必须要有
        super(LinearModel, self).__init__()
        # torch.nn.Linear是pytorch的一个类
        # 继承自nn.Module 能够自动进行反向传播
        # 类后面加括号实际就是在构造对象
        # Linear这个类包含两个member Tensor:
        # weight and bias
        # 可以直接自动实现权重乘以输入加上偏置
        # 所以self.linear就是一个nn.Linear类型的类
        # Linear就是nn的一个组件
        # 这个linear是callable的
        self.linear = torch.nn.Linear(1, 1)

    # 必须是forward()这个名字，应该是接口函数
    # 在前馈过程中需要进行的计算
    # 没有backward()是因为用Module构造出来的对象
    # 这种对象会自动的根据这里面的计算图自动实现
    # backward()的过程 由Module来自动完成的 可构建计算图
    # 在pytorch无法进行我特制模块的求导计算时候
    # 比如效率不高这种情况 有更高效方法时候
    # 可以从Functions这个类进行继承（这个类需要实现
    # 反向传播的 需要自己特殊定制的计算块
    # 当然有现成的就用现成的 不用自己另设计反向传播的模块
    # Module类中应该有__call__()方法，这个方法中定义了forawrd()
    # 所以必须要实现这个forwar()方法并覆盖
    def forward(self, x):
        # 在对象后面加一个括号就是实现一个可调用的对象
        y_pred = self.linear(x)
        return y_pred

# 这个model是callable的
# 可以使用model(x)将x送到forward()中进行计算
model = LinearModel()

#################################
# *args和**kargs分别压缩元组和字典
# 比如func(1, 2, 3, x=4, y=5)
# 那么可以定义func(*args, **kargs)
# *args里面是(1, 2, 3)
# **kargs里面则是{x: '4', y: '5'}
#################################

# 构造损失函数和优化器
# nn.MSELoss也是继承自nn.Module
# size_average这个参数被弃用了 应该改为reduction
# criterion = torch.nn.MSELoss(size_average=False)
# size_average=True对应于reduction='mean'
criterion = torch.nn.MSELoss(reduction='sum')

# 优化器不是Module 不会构建计算图
# 第一个参数是待优化参数 调用parameters()函数把model里的
# 所有待优化参数找出来进行优化
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training Cycle
# 训练过程：
#   -求y_hat
#   -求loss 这两步就是forward
#   -清零并进行backward
#   -更新
# 总结起来就是前馈、反馈、更新-->前馈、反馈、更新-->......
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # 注意这里的loss是一个对象，这个对象包含__str__()函数
    # 所以它不会产生计算图 打印的时候就是一个标量
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward() # Backward: Autograd
    # Updata step()函数就是进行一次更新
    # 会根据所有参数包含的梯度以及预先设置的学习率
    # 进行自动更新
    optimizer.step() 

# Output weight and bias
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

# Test Model
x_test = torch.Tensor([[4.0]]) # 1X1的矩阵
y_test = model(x_test)
print('y_pred = ', y_test.data)