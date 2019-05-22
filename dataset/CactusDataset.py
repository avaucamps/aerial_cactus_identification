import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CactusDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            train (boolean): Whether to load the train set or validation set.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cactus_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    
    def __len__(self):
            return len(self.cactus_data)

        
    def __getitem__(self, index):
            img_name = os.path.join(
                self.root_dir, 
                self.cactus_data.iloc[index, 0]
            )

            image = Image.open(img_name)
            label = self.cactus_data.iloc[index, 1]

            if self.transform:
                image = self.transform(image)

            return image, label