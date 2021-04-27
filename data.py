import os

import numpy as np
from PIL import Image
import torch
import torch.nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
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

        self.person_data = person_model_input(self.obs, self.train_steps) #(num_obs, 8, 2)      
        self.group_data = log_group_model_input(self.obs, self.train_steps, neighborhood_size, image_size_dims, neighborhood_radius, grid_radius, grid_angle, [1, 1, 1, 1, 1, 1, 1, 1], self.raw_data) #(num_obs, 8, -1)
        # self.scene_data = None #(num_obs, 3, 720, 576)

        # apply transformations
        self.transformations = transforms.Compose([
            transforms.ToTensor()
        ])
        self.ground_truth = model_expected_ouput(self.pred, self.pred_steps)
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, index):
        sample = self.obs[index] # 1x8x4
        frame_ID = sample[-1][1] # gets the frame ID
        img = self.load_image(self.image_dir, 'seq_hotel/frames', frame_ID)

        img = self.transformations(img)
        
        
        input_data = {}
        input_data['person_data'] = self.person_data[index]
        input_data['group_data'] = self.group_data[index]
        input_data['image']  = img       
        return input_data

    def load_image(self, data_dir, subset, frame_ID):
        '''
        args:
            data_dir - data root directory
            subset - name of data subset: 'seq_eth/frames' or 'seq_hotel/frames'
            frame_ID - id belonging to obs file
            
        out:
            img np.array
        '''
        if subset == 'seq_hotel/frames':
            assert ((frame_ID <= 18060) and (frame_ID >= 0))
        if subset == 'seq_eth/frames':
            assert ((frame_ID <= 12381) and (frame_ID >= 780))

        _img_ext = '-U.png'
        img_path = os.path.join(data_dir, subset, str(frame_ID + 1) + _img_ext)
        img = np.array(Image.open(img_path))

        # assert list(img.shape[:2]) == self.image_size_dims[::-1]

        return img

