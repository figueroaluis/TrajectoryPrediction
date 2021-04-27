import torch
import torch.nn
from torch.utils.data import Dataset
import numpy as np
from utils import person_model_input, log_group_model_input, model_expected_ouput, preprocess

class BaseDataset(Dataset):
    def __init__(self, obs_path, pred_path, raw_path, image_dir, image_size_dims = [720, 576],  neighborhood_size = 32, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45):
        super().__init__()
        self.obs = np.load(obs_path) # obs_path should combine files from all sources or we should provide separate paths for separate datasets
        self.pred = np.load(pred_path)

        self.raw_data = preprocess(raw_path)

        self.image_dir = image_dir
        self.train_steps = 8
        self.pred_steps = 12
        #TODO: call utils api to split data into person, group, scene data        

        self.person_data = person_model_input(self.obs, self.train_steps)        
        self.group_data = log_group_model_input(self.obs, self.train_steps, neighborhood_size, image_size_dims, neighborhood_radius, grid_radius, grid_angle, [1, 1, 1, 1, 1, 1, 1, 1], self.raw_data)
        self.scene_data = None

        self.ground_truth = model_expected_ouput(self.pred, self.pred_steps)
    
    def __len__(self):
        return len(self.raw_obs)
    
    def __getitem__(self, index):
        

