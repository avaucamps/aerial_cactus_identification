import torch

def get_device():
    if torch.cuda.is_available():
        print("Using cuda:0")
        return torch.device("cuda:0")
    
    print("Using cpu")
    return torch.device('cpu')