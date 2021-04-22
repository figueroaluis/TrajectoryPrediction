import torch
import torch.nn
from torch.utils.data import Dataset
import numpy as np
from utils import *

class BaseDataset(Dataset):
    def __init__(self, raw_obs_path, image_dir):
        super().__init__()
        self.raw_obs = np.load(raw_obs_path)
        self.image_dir = image_dir
        self.train_steps = 8
        self.pred_steps = 12
        #TODO: call utils api to split data into person, group, scene data
        self.person_data = 
        self.group_data = None
        self.scene_data = None
    
    def __len__(self):
        return len(self.raw_obs)
    
    def __getitem__(self, index):
