import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data.cifar import CIFAR20


class SampleDataSet(Dataset):

    ''' samples is a np.ndarray representation of a list of images '''
    def __init__(self, samples, targets=None, labels=None, transform=None):

        super(SampleDataSet, self).__init__()
        self.BASE_DATASET = CIFAR20(train=True, transform=transform, download=True)
        self.SAMPLE_DATA = samples
        if targets == None and labels == None: 
            self.SAMPLE_TARGETS = [20]*samples.shape[0]
        elif targets == None:
            self.SAMPLE_TARGETS = [self.classes.index(lbl) for lbl in labels]
        else:
            self.SAMPLE_TARGETS = targets
        
        self.classes = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 
                        'household electrical devices', 'househould furniture', 'insects', 'large carnivores', 
                        'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores', 
                        'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 
                        'trees', 'vehicles 1', 'vehicles 2','unknown']
        
        self.transform = transform
        self.data = np.concatenate([self.BASE_DATASET.data,self.SAMPLE_DATA])
        self.targets = np.concatenate([self.BASE_DATASET.targets,self.SAMPLE_TARGETS])
        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def get_sample(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.SAMPLE_DATA[index], self.SAMPLE_TARGETS[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]        

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 
               'index': index, 'class_name': class_name}}
        
        return out

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
    
    def get_sample_image(self, index):
        img = self.SAMPLE_DATA[index]
        return img

    def sample_len(self):
        return len(self.SAMPLE_DATA)

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: Single Image Dataset"