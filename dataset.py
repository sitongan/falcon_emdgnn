import glob
import os
from torch_geometric.data import (Data, Dataset)
import torch

class FalconDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(FalconDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            self.processed_files = glob.glob(os.path.abspath('data/processed') + '/*.pt')
        return self.processed_files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data