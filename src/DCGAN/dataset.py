from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

'''
Reusable class for processing the dataset of the image.
'''
class MonetDataSet(Dataset):
    '''
    Constructor for the class.
    '''
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)
        self.transform = transform
    
    '''
    Returns number of samples in the dataset.
    '''
    def __len__(self):
        return len(self.file_names)
    
    '''
    Returns a specific item of the dataset.
    '''
    def __getitem__(self, index):
        img_path = os.path.join(self.folder_path, self.file_names[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
'''
Wrapper class around MonetDataset to load data with custom attributes to Pytorch's Data Loader class.
'''
class MonetDataLoader():
    '''
    Constructor to initialize the attributes.
    '''
    def __init__(self, folder_path, transform=None, batch_size = 32, shuffle = True):
        self.folder_path = folder_path
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    '''
    Function to get the dataloader.
    '''
    def get_data_loader(self):
        return DataLoader(MonetDataSet(folder_path=self.folder_path, transform=self.transform),
                          batch_size=self.batch_size,
                          shuffle=self.shuffle)
    