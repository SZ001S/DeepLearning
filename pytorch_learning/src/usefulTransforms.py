from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open('../datasets/train/ants_image/0013035.jpg')
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize - 2 能将好几个transforms的操作进行打包
# Compose()中的参数需要是一个列表 其中列表的元素类型是transform类型
# 所以得到Compose([transforms参数1, transforms参数2, ...])
# 仅仅是一个整数就是将尺寸调整至最小边尺寸 回忆下我是如何在word中插入图片的
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)


# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop((500, 600))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    # 将transforms类型的操作打包
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)






writer.close()

