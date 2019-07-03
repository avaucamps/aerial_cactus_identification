from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import utils, models
from dataset.CactusDataset import CactusDataset
from tqdm import tqdm
import torch
from utils import get_device
from constants import MODEL_PATH
from path_constants import TEST_DATA_PATH, SAMPLE_SUBMISSION_CSV, SUBMISSION_CSV
from torch.utils.data import dataset
from torchvision import transforms, datasets
import pandas as pd
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Test:
    def __init__(self):
        self.device = get_device()
        self._load_data()
        self._load_model()

    
    def _load_model(self):
        self.model = torch.load(MODEL_PATH)
        self.model.eval()


    def run(self):
        predict = []
        self.model.eval()
        for i, (data, _) in enumerate(tqdm(self.test_loader)):
            data = data.cuda()
            output = self.model(data)

            pred = torch.sigmoid(output)
            predicted_vals = pred > 0.5
            predict.append(int(predicted_vals))

        self._create_submission_file(predict)


    def _load_data(self):
        test_data = CactusDataset(
            csv_file = SAMPLE_SUBMISSION_CSV, 
            root_dir = TEST_DATA_PATH, 
            transform = transforms.Compose([transforms.ToTensor()])
        )
        self.test_loader = DataLoader(dataset = test_data, shuffle = False)


    def _create_submission_file(self, predict):
        submit = pd.read_csv(SAMPLE_SUBMISSION_CSV)
        submit['has_cactus'] = predict
        submit.to_csv(SUBMISSION_CSV, index=False)