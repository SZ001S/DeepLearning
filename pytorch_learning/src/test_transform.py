from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
# transforms.py是一个工具箱 能将比如图片 通过totensor，resize等工具 产生一个结果
# python -> tensor数据类型
# transforms.ToTensor()创建具体的工具


img_path = '../datasets/train/ants_image/0013035.jpg'
img = Image.open(img_path)

writer = SummaryWriter('logs')

tensor_trains = transforms.ToTensor()
tensor_img = tensor_trains(img)
# print(tensor_img)

writer.add_image('Tensor_img', tensor_img)

writer.close()
