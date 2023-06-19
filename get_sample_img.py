from data.cifar import CIFAR20
from random import randint

DATASET = CIFAR20(train=False, transform=None, download=True)

def get_pic(num=None):
    if num==None: num = randint(0,len(DATASET)-1)
    pic = DATASET[num]['image']
    data = DATASET.get_image(num)
    return (num,pic,data)

def main():
    num,pic,_ = get_pic()
    pic.save("./data/cifar_img_%d.jpeg"%num)

if __name__ == "__main__":
    main() 