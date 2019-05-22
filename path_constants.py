import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = os.path.join(BASE_PATH, "data/train")
TRAIN_ALL_CSV = os.path.join(BASE_PATH, "data/train_all.csv")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "data/train.csv")
VALIDATION_CSV_PATH = os.path.join(BASE_PATH, "data/validation.csv")