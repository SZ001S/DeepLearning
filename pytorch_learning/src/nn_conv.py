import torch
import torch.nn.functional as F

inputs = torch.tensor([[1, 2, 0, 3, 1],
                       [0, 1, 2, 3, 1],
                       [1, 2, 1, 0, 0],
                       [5, 2, 3, 1, 1],
                       [2, 1, 0, 1, 1]])  # 尺寸不满足要求

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])  # 同样不满足要求

inputs = torch.reshape(inputs, (1, 1, 5, 5))  # minibatch_size, channels, height, width
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(inputs.shape)
print(kernel.shape)

outputs = F.conv2d(inputs, kernel, stride=1)
print(outputs)

outputs_2 = F.conv2d(inputs, kernel, stride=2)
print(outputs_2)

outputs_3 = F.conv2d(inputs, kernel, stride=1, padding=1)
print(outputs_3)
