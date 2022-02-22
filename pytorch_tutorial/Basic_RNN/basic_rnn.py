# RNN用于处理带有序列形式的数据
import torch

# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
# num_layers = 1


# # cell = torch.nn.RNNCell(
# #     input_size=input_size, 
# #     hidden_size=hidden_size
# # )

# cell = torch.nn.RNN(
#     input_size=input_size, 
#     hidden_size=hidden_size,
#     num_layers=num_layers
# )

# # (seq, batch, feature)
# # dataset = torch.randn(seq_len, batch_size, input_size)
# # hidden = torch.zeros(batch_size, hidden_size)

# inputs = torch.randn(seq_len, batch_size, input_size)
# hidden = torch.zeros(num_layers, batch_size, hidden_size)

# # for index, input in enumerate(dataset):
# #     print('='*20, index, '='*20)
# #     print('Input size: ', input.shape)

# #     hidden = cell(input, hidden)

# #     print('Output size: ', hidden.shape)
# #     print(hidden)

# out, hidden = cell(inputs, hidden)

# print('Output size:', out.shape)
# print('Output:', out)
# print('Hidden size:', hidden.shape)
# print('Hidden:', hidden)


# Train a model to leran: 'hello' -> 'ohlol'

# Prepare Data
input_size = 4
hidden_size = 4
batch_size = 1

# 打印输出结果用
idx2char = ['e' , 'h', 'l', 'o']
# hello
x_data = [1, 0, 2, 2, 3]
# ohlol
y_data = [3, 1, 2, 3, 2]

# 查询
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
# 表示hello的x_one_hot                  
x_one_hot = [one_hot_lookup[x] for x in x_data]

# Reshape the inputs to
# (seqLen, batchSize, inputSize)
inputs = torch.Tensor(x_one_hot).view(
    -1, batch_size, input_size
)
# Teshape the labels to 
# (seqLen, 1)
labels = torch.LongTensor(y_data).view(-1, 1)

# Design Model
class Model(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, batch_size
    ) -> None:
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnncell = torch.nn.RNNCell(
            input_size=self.input_size, 
            hidden_size=self.hidden_size
        )

    # 将rnncell的输出作为下一个rnncell单元的输入
    # h(t) = rnncell(x(t), h(t-1))
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden
    
    # 生成初始隐层h(0)
    def init_hidden(self):
        return torch.zeros(
            self.batch_size,
            self.hidden_size
        )


device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
net = Model(input_size, hidden_size, batch_size)
net.to(device)

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)


# Training Cycle
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()

    # 求h(0)
    hidden = net.init_hidden().to(device)
    print('Predicted string: ', end='')
    # inputs: (seqLen, batchSize, inputSize)
    # input: (batchSize, inputSize)
    # labels: (seqSize, 1)
    # label: (1)
    for input, label in zip(inputs, labels):
        input, label = input.to(device), label.to(device)
        # 这里调用net中的forward()方法
        hidden = net(input, hidden)
        # 因为求的loss是一个总和 
        # 需要构造计算图 不要使用item()
        loss += criterion(hidden, label)

        # Output prediction
        # hidden是一个4x1的向量 
        # 分别表示e, h, l, o
        # 这里返回一个值与下标的捆绑元组
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    
    loss.backward()
    optimizer.step()
    print(
        f', Epoch [{(epoch+1)}/15] loss={loss.item():.4f}'
    )