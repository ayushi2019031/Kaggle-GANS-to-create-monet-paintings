import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MonetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root folder with trainA, trainB, testA, testB subfolders
            transform (callable, optional): Optional transform to be applied on images.
            mode (str): One of ['train', 'test']
        """
        self.root_dir = root_dir
        self.transform = transform

        self.dir_A = os.path.join(root_dir, f"monet_jpg")
        self.dir_B = os.path.join(root_dir, f"photo_jpg")

        self.files_A = sorted(os.listdir(self.dir_A))
        self.files_B = sorted(os.listdir(self.dir_B))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A_path = os.path.join(self.dir_A, self.files_A[idx % len(self.files_A)])
        img_B_path = os.path.join(self.dir_B, self.files_B[idx % len(self.files_B)])

        img_A = Image.open(img_A_path).convert("RGB")
        img_B = Image.open(img_B_path).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

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
        return DataLoader(MonetDataset(root_dir=self.folder_path, transform=self.transform),
                          batch_size=self.batch_size,
                          shuffle=self.shuffle)
    