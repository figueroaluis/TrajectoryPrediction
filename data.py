import os
import numpy as np
from PIL import Image
import torch
import torch.nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from utils import person_model_input, log_group_model_input, model_expected_ouput, preprocess
import pdb

class BaseDataset(Dataset): #TODO: image_dir is still single path
    def __init__(self, datasets_path_list, image_size_dims = [720, 576],  neighborhood_size = 32, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45, train_steps = 5, pred_steps = 5):
        super().__init__()                
        
        self.train_steps = train_steps
        self.pred_steps = pred_steps

        person_data = []
        group_data = []
        scene_data = []
        ground_truth = []
        for i, dataset_path in enumerate(datasets_path_list):
            obs = np.load(dataset_path + "/obs.npy") 
            print(dataset_path, ": {}".format(len(obs)))
            ground_truth.append((model_expected_ouput(np.load(dataset_path + "/pred.npy"), self.pred_steps)))
            raw_data,_ = preprocess(dataset_path + "/pixel_pos.csv")

            person_data.append(person_model_input(obs, self.train_steps))            
            group_data.append(log_group_model_input(obs, self.train_steps, neighborhood_size, image_size_dims, neighborhood_radius, grid_radius, grid_angle, [1, 1, 1, 1, 1, 1, 1, 1], raw_data))

            scene_data += self.get_sceneData(obs, dataset_path)
            

        self.person_data = np.concatenate(person_data, axis = 0) #(num_obs, 8, 2)      
        self.group_data = np.concatenate(group_data, axis = 0) #(num_obs, 8, -1)
        self.scene_data = scene_data #(num_obs, 3, 720, 576)
        self.ground_truth = np.concatenate(ground_truth, axis = 0) #(num_obs, 12, 2)
      
        #TODO: call utils api to split data into person, group, scene data

        # apply transformations
        self.transformations = transforms.Compose([
            transforms.ToTensor()
        ])
        
    
    def __len__(self):
        return len(self.person_data)
    
    def __getitem__(self, index):
      
        input_data = {}
        input_data['person_data'] = self.person_data[index]
        input_data['group_data'] = self.group_data[index]
        input_data['scene_data']  = self.transformations(self.scene_data[index])  
        input_data['gt'] = self.ground_truth[index] 
        return input_data

    def load_image(self, data_dir, frame_ID):
        '''
        args:
            data_dir - data root directory
            subset - name of data subset: 'seq_eth/frames' or 'seq_hotel/frames'
            frame_ID - id belonging to obs file
            
        out:
            img np.array
        '''
        # if subset == 'seq_hotel/frames':
        #     assert ((frame_ID <= 18060) and (frame_ID >= 0))
        # if subset == 'seq_eth/frames':
        #     assert ((frame_ID <= 12381) and (frame_ID >= 780))

        _img_ext = '-U.png'

        img_path = os.path.join(data_dir, str(int(frame_ID)) + _img_ext)
        img = np.array(Image.open(img_path))

        # assert list(img.shape[:2]) == self.image_size_dims[::-1]

        return img

    def get_sceneData(self, obs,base_dir):
        scene_images = []
        
        image_dir = os.path.join(base_dir, 'frames')
        for ob in obs:
            frame_ID = ob[-1][1] # gets the frame ID
            scene_images.append(self.load_image(image_dir, frame_ID))
        return scene_images

