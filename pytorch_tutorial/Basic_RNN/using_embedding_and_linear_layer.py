# RNN用于处理带有序列形式的数据
import torch
device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
# Train a model to leran: 'hello' -> 'ohlol'

# Prepare Data
# parameters
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5


# 打印输出结果用
idx2char = ['e' , 'h', 'l', 'o']
# hello
x_data = [[1, 0, 2, 2, 3]] # (batch, seq_len)
# ohlol
y_data = [3, 1, 2, 3, 2] # (batch * seq_len)


inputs = torch.LongTensor(x_data) # (batchSize, seqLen)
labels = torch.LongTensor(y_data) # (batchSize * seqLen)

# Design Model
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        # Lookup matrix of Embedding:
        # (inputSize, embeddingSize)
        self.emb = torch.nn.Embedding(
            input_size, embedding_size
        )
        # Input of RNN:
        # (batchSize, seqLen, embbedingSize)
        # Output of RNN
        # (batchSize, seqLen, hiddenSize)
        self.rnn = torch.nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # Input of FC Layer:
        # (batchSize, seqLen, hiddenSize)
        # Output of FC Layer:
        # (batchSize, seqLen, numClass)
        self.fc = torch.nn.Linear(
            hidden_size, num_class
        )

    def forward(self, x):
        hidden = torch.zeros(
            num_layers,
            x.size(0), hidden_size
        ).to(device)
        # Input should be LongTensor长整型张量 (batchSize, seqLen)
        # Output with shape:
        # (batchSize, seqLen, embeddingSize)
        # Notice: batch FIRST
        x = self.emb(x) # (batch, seqLen, embeddingSize)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        # Reshape result to use Cross Entropy Loss:
        # (batchSize x seqLen, numClass)
        return x.view(-1, num_class)
    

net = Model()
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

