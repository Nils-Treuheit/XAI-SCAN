from data.cifar import CIFAR20
from random import randint
from PIL import Image


dataset = CIFAR20(train=False, transform=None, download=True)
num = randint(0,len(dataset)-1)
pic = Image.fromarray(dataset.get_image(num))
pic.save("./data/cifar_img_%d.jpeg"%num)
