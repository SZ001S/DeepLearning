import torch

# 将父类模板拿过来(继承)修改使用
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input):
        output = input + 1
        return output

model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)
