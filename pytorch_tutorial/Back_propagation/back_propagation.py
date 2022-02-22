import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True #需要计算梯度的，打个标签


# model 定义数学操作 构建计算图
def forward(x):
    return x * w


#没调用一次loss函数就是构建一次动态图
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('Predict (befor training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 这些地方需要注意，在更新权重时候
        # 只是数值上的修改，不能涉及到计算图
        # 构建，所以进行一些转换，使得求出
        # 梯度不参与计算图构建之中
        # Tensor参与计算时会在构建计算图
        # 这是一个很重要的一点
        l = loss(x, y)
        # l是张量Tensor 它有一个成员函数backward()
        # 然后就会自动的把计算图上的所有需要计算梯度的地方
        # 都求出来，并存放在变量里面(这里就是w)，存放之后
        # 计算图就会被释放了，只要用一次backward()函数就会
        # 释放一次，再进行一次loss计算时，再次创建一个
        # 新的计算图，在构建实际的神经网络时可能在运行时会有
        # 计算图不一样的情况（保持灵活，多样性）这是pytorch的
        # 的核心竞争力 
        l.backward()
        # item()函数是把w.grad中的数值拿出来变成python中
        # 的标量，w.grad就一个值，将其变为标量，这也是
        # 防止产生计算图
        print('\tgrad: ', x, y, w.grad.item())
        # 这里计算出来的w也是一个Tensor 一个Tensor
        # 包含data和grad这两个成员变量
        # 使用w.data防止又一次构建计算图
        # 并不是希望修改w的grad时候还要求梯度（纯数值修改）
        # 不涉及模型 这里的w.grad跟套娃似的也是一个Tensor
        w.data = w.data - 0.01 * w.grad.data

        # 这里又是一个重要点
        # 需要把权重清零，防止进行下一次权重更新时候
        # 把原来的权重也顺带加上了，清零就会加零
        # 这一步也不是必须，需要具体模型具体看待
        # 所以这一步操作从设计上被解耦出来供使用者酌情使用

        w.grad.data.zero_()
    
    # 这里的l.item()就是l的python显示精度很高数值形式
    print('prograss: ', epoch, l.item())

print('predict (after training)', 4, forward(4).item())