import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class SampleDataSet(Dataset):

    def __init__(self, samples, targets=None, labels=None, transform=None):

        super(SampleDataSet, self).__init__()
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'unknown']
        self.transform = transform
        self.data = samples
        if targets == None and labels == None: 
            self.targets = [-1]*samples.shape[0]
        elif self.targets == None:
            self.targets = [self.classes.index(lbl) for lbl in labels]
        else:
            self.targets = targets
        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]        

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 
               'index': index, 'class_name': class_name}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Single Image Dataset"