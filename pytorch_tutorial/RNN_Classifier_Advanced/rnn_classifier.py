# Name Classification
import enum
import time
import torch
import gzip
import csv
from torchvision import transforms
# from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# 修改部分
from torch.nn.utils.rnn import pack_padded_sequence
#-------------------------------
# 会用到relu函数
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='0'
print(torch.cuda.device_count())


# Preparing Data
# 数据集
# Python中的super()方法设计目的是用来解决多重继承时父类的查找问题，
# 所以在单重继承中用不用 super 都没关系；
# 但是，使用 super() 是一个好的习惯。
# 一般我们在子类中需要调用父类的方法时才会这么用。
class NameDataset(Dataset):
    # super()用法；
    # https://blog.csdn.net/qq_14935437/article/details/81458506
    def __init__(self, is_train_set=True) -> None:
        filename = r'../课件/PyTorch深度学习实践/names_train.csv.gz' \
            if is_train_set else \
                r'../课件/PyTorch深度学习实践/names_test.csv.gz'
        
        # 文本文件中的 回车 在不同操作系统中所用的字符表示有所不同。
        # Windows:
        # \r\n
        # Linux/Unix:
        # \n
        # Mac OS:
        # \r

        # python读写文件 open()中
        # r rb rt
        # rt模式下，python在读取文本时会自动把\r\n转换成\n.

        # 使用’r’一般情况下最常用的，但是在进行读取二进制文件时，可能会出现文档读取不全的现象。
        # 使用’rb’按照二进制位进行读取的，不会将读取的字节转换成字符
        # 二进制文件就用二进制方法读取’rb’
        # r为仅读取 w为仅写入 a为仅追加

        # r+为可读写两种操作 w+为可读写两种操作（会首先自动清空文件内容） a+为追加读写两种操作
        # 以上三种操作方式均不可同时进行读写操作
        # ————————————————
        # 版权声明：本文为CSDN博主「凌晨点点」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        # 原文链接：https://blog.csdn.net/lhh08hasee/article/details/79354963
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len
    
    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]

    def getCountriesNum(self):
        return self.country_num

# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2
N_CHARS = 128
N_EPOCHS = 100
USE_GPU = True

# 实例化数据集对象
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# N_COUNTRY is the output size of our model
N_COUNTRY = trainset.getCountriesNum()


# Model Design
class RNNClassifier(torch.nn.Module):
    def __init__(
        self, input_size, 
        hidden_size, 
        output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(
            hidden_size, hidden_size, n_layers, 
            bidirectional=bidirectional
        )

        self.fc = torch.nn.Linear(
            hidden_size * self.n_directions, 
            output_size
        )

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(
            self.n_layers * self.n_directions,
            batch_size, self.hidden_size
        )
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input shape : B x S -> S x B
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        # pack them up
        # 对于一些列需要插值保证形状一致为一个正常矩阵
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            # dim=0代表是列，dim=1代表是行
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output
    

# We have to sort the batch element by length of sequence
def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)

# 使用GPU的操作封装成函数
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        tensor = tensor.to(device)
    return tensor


def make_tensors(names, countries):
    sequence_and_lengths = [name2list(name) for name in names]
    # 取出所有的列表中每个姓名的ASCII码序列
    name_sequences = [s1[0] for s1 in sequence_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequence_and_lengths])
    countries = countries.long()

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    
    # sort by length to use torch._pack_padded_sequence()
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths), \
           create_tensor(countries)


def trainModel():
    total_loss = 0
    # 注意这里从1开始枚举
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        # 1. forward - compute output of model
        # 2. forward - comput loss
        # 3. zero grad
        # 4. backward
        # 5. update
        # 注意缩进
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f'[{time_since}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        percent = f'{100 * correct / total : .2f}'
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    
    return correct / total

# main cycle
def time_since(since):
    s = time.time() - since
    m = np.math.floor(s / 60)
    s -= m * 60
    return f'{m:d}m {s:d}s'

if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)

    if USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        classifier.to(device)

    # criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print(f'Training for {N_EPOCHS}')
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        # Train cycle
        trainModel()
        acc = testModel()
        acc_list.append(acc)
    
    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()