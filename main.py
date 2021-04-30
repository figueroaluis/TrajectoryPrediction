import torch
import torch.nn as nn
import numpy as np
import argparse

from tensorboardX import SummaryWriter
from data import BaseDataset
from torch.utils.data import DataLoader
import os
from utils import DataProcesser
from model import Model

def validate(model, val_dataloader):
    
    for batch_id, data in enumerate(val_dataloader):

        data['person_data'] = data['person_data'].cuda().float()
        data['group_data'] = data['group_data'].cuda().float()
        data['scene_data'] = data['scene_data'].cuda().float()
        data['gt'] = data['gt'.cuda()]

        op = model(data['scene_data'], data['group_data'], data['person_data'])
    

def train(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs):
    writer = SummaryWriter()
    model.train()
    for epoch in range(num_epochs):
        
        for batch_id, data in enumerate(train_dataloader):

            data['person_data'] = data['person_data'].float().cuda()
            data['group_data'] = data['group_data'].float().cuda()
            data['scene_data'] = data['scene_data'].float().cuda()
            data['gt'] = data['gt'].float().cuda()
            optimizer.zero_grad()

            op = model(data['scene_data'], data['group_data'], data['person_data'])
            loss = criterion(op,data['gt'])
            loss.backward()
            optimizer.step()
        # model.eval()
        # validate(model , val_dataloader)
            print("Epoch: " + str(epoch), " Batch: " + str(batch_id) + " Loss: " + str(loss.item()))
            writer.add_scalar('Loss/train', loss,batch_id)
    writer.close()
if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load hyperparams')    
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)    
    parser.add_argument('--lr', type=float, default=4e-4)    
    args = parser.parse_args()

    datasets_names_list = [] #TODO: add dataset names here

    for i, data_path in enumerate(datasets_names_list):  
        datasets_names_list[i] = args.data_dir +"/"+ data_path
        data_path = args.data_dir +"/"+ data_path  
        if not os.path.exists(data_path + "/obs.npy"):                    
            dp = DataProcesser(data_path, 8, 12)
            dp.save_files(data_path)        

    epochs = args.num_epochs
    train_dataset = BaseDataset(datasets_names_list, args.data_dir+"/frames/") # TODO: frames is still single path
    trainloader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers = args.num_workers)    

    model = Model()
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay = 1e-5)

    
    train(model,criterion, optimizer, trainloader, None, 10)

    