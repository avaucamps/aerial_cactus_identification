from enum import Enum

class ModelType(Enum):
    cnn = 1
    fine_tuned = 2
    fixed_feature_extractor = 3