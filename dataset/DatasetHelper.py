import pandas as pd
from path_constants import *
from dataset.CactusDataset import CactusDataset
from skimage import io, transform
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class DatasetHelper:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._prepare_dataset()


    def load_dataset(self):
        """
        Loads dataset from CSV files and put it in DataLoader objects.

        Returns:
            train_loader, validation_loader: DataLoader object for training and DataLoader object
            for validation.
        """
        train_dataset = CactusDataset(
            csv_file = TRAIN_CSV_PATH,
            root_dir = TRAIN_DIR,
            transform = self._get_transforms()
        )
        validation_dataset = CactusDataset(
            csv_file = VALIDATION_CSV_PATH,
            root_dir = TRAIN_DIR,
            transform = self._get_transforms()
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_loader, validation_loader


    def _prepare_dataset(self):
        """
        If there is no csv file for training set and validation set, creates them from the csv file containing the
        whole dataset.
        """
        if os.path.isfile(TRAIN_CSV_PATH) and os.path.isfile(VALIDATION_CSV_PATH):
            return

        dataset = pd.read_csv(TRAIN_ALL_CSV)
        columns = dataset.columns
        X = dataset.loc[:, columns[0]].values
        y = dataset.loc[:, columns[1]].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        train_df = pd.DataFrame(
            list(zip(X_train, y_train)), columns=[columns[0], columns[1]]
        )
        valid_df = pd.DataFrame(
            list(zip(X_test, y_test)), columns=[columns[0], columns[1]]
        )
        train_df.to_csv(TRAIN_CSV_PATH, index=False)
        valid_df.to_csv(VALIDATION_CSV_PATH, index=False)


    def _get_transforms(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])