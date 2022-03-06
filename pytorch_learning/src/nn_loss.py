import torch

inputs = torch.tensor([1., 2., 3.])
targets = torch.tensor([1., 2., 5.])

# batch_size channel height width
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

print(inputs, targets)

loss = torch.nn.L1Loss(reduction='mean')
result = loss(inputs, targets)

loss_mse = torch.nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)
