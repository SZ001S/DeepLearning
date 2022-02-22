# Name Classification
import torch
import gzip
import csv
from torchvision import transforms
# from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#-------------------------------
# 会用到relu函数
import torch.nn.functional as F
import torch.optim as optim


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
        filename = '课件\PyTorch深度学习实践\names_train.csv.gz' \
            if is_train_set else \
                '课件\PyTorch深度学习实践\names_test.csv.gz'
        
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
        pass

    def idx2country(self, index):
        pass

    def getCountriesNum(self):
        pass