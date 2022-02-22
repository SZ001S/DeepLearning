# RNN用于处理带有序列形式的数据
import torch
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
# Train a model to leran: 'hello' -> 'ohlol'

# Prepare Data
input_size = 4
hidden_size = 4
num_layers = 1
batch_size = 1
seq_len = 5

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
    seq_len, batch_size, input_size
)
# Teshape the labels to 
# (seqLen x batchSize, 1)
labels = torch.LongTensor(y_data)

# Design Model
class Model(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, batch_size, 
        num_layers=1
    ) -> None:
        super(Model, self).__init__()
        # input_size = 4
        # hidden_size = 4
        # num_layers = 1
        # batch_size = 1
        # seq_len = 5
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = torch.nn.RNN(
            input_size=self.input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

    def forward(self, input):
        # batch_size为了构造隐层中的 
        # h(0) 
        # hidden: (numLayers, batchSize, hiddenSize)
        hidden = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.hidden_size
        ).to(device)
        out, _ = self.rnn(input, hidden)
        # out: (seqLen x batchSize, hiddenSize)
        return out.view(-1, self.hidden_size)
    

net = Model(input_size, hidden_size, batch_size, num_layers)
net.to(device)

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)


# Training Cycle
for epoch in range(15):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    # 不是idx = idx.data.numpy()
    idx = idx.cpu().numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(
        f', Epoch [{epoch+1}/15] loss = {loss.item():.3f}'
    )
    ########################################
    # 维度诅咒 诅咒你不能完成训练 
    # 除非你有足够的样本 用来训练采样
    # 而每增加一个维度 样本数量增加却是指数级的
    ########################################

    #######################
    # One-hot vector: 
    # High-dimention
    # Sparse
    # Hardcoded
    #######################
    # 数据降维
    # Embedding vectors: 
    # Lower-dimension
    # Dense
    # Learned from data
    #######################

