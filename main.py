import torch
import torch.nn as nn
import numpy as np
import argparse
import pdb
import sys
from scipy.spatial import distance
import heapq

from tensorboardX import SummaryWriter
from data import BaseDataset
from torch.utils.data import DataLoader
import os
from utils import DataProcesser
from model import Model

def validate(model, val_dataloader):
    mean_FDE = 0
    mean_ADE = 0
    all_FDE = 0
    all_ADE = 0
    total = 0
    model.eval()
    for batch_id, data in enumerate(val_dataloader):
        data['person_data'] = data['person_data'].cuda().float()
        data['group_data'] = data['group_data'].cuda().float()
        data['scene_data'] = data['scene_data'].cuda().float()
        op = model(data['scene_data'], data['group_data'], data['person_data']).detach().cpu()
        mean_FDE += calculate_FDE(data['gt'], op, len(data['gt']), 1)        
        mean_ADE += calculate_ADE(data['gt'], op, len(data['gt']), 5, 1)
        all_FDE += calculate_FDE(data['gt'], op, len(data['gt']), len(data['gt']))
        all_ADE += calculate_ADE(data['gt'], op, len(data['gt']), 5, len(data['gt']))
        total += 1
    print("mean FDE: {}".format(mean_FDE/total))
    print("mean ADE: {}".format(mean_ADE/total))
    print("all FDE: {}".format(all_FDE/total))
    print("all ADE: {}".format(all_ADE/total))

def train(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs):
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        model.train()
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
            print("Epoch: " + str(epoch), " Batch: " + str(batch_id) + " Loss: " + str(loss.item()))
            writer.add_scalar('Loss/train', loss,batch_id)
        if epoch % 2 == 0:
            validate(model , val_dataloader)

    writer.close()
    
def calculate_FDE(test_label, predicted_output, test_num, show_num):
    total_FDE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        total_FDE[i] = distance.euclidean(predicted_result_temp[-1], label_temp[-1])

    show_FDE = heapq.nsmallest(show_num, total_FDE)
    show_FDE = np.reshape(show_FDE, [show_num, 1])
    return np.average(show_FDE)


def calculate_ADE(test_label, predicted_output, test_num, predicting_frame_num, show_num):
    total_ADE = np.zeros((test_num, 1))
    for i in range(test_num):
        predicted_result_temp = predicted_output[i]
        label_temp = test_label[i]
        ADE_temp = 0.0
        for j in range(predicting_frame_num):
            ADE_temp += distance.euclidean(predicted_result_temp[j], label_temp[j])
        ADE_temp = ADE_temp / predicting_frame_num
        total_ADE[i] = ADE_temp

    show_ADE = heapq.nsmallest(show_num, total_ADE)
    show_ADE = np.reshape(show_ADE, [show_num, 1])
    return np.average(show_ADE)

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load hyperparams')    
    parser.add_argument('--data_dir', type=str, default="Data")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_steps', type=int, default=5)
    parser.add_argument('--test_steps', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--cnn', type=str, default="baseline")
    args = parser.parse_args()
    datasets_names_list = ["seq_eth", "seq_hotel", "zara02"] #TODO: add dataset names here
    
    for i, data_path in enumerate(datasets_names_list):  
        datasets_names_list[i] = args.data_dir +"/"+ data_path
        data_path = args.data_dir +"/"+ data_path  
        if not os.path.exists(data_path + "/obs.npy"): 
            print(data_path, "NOT THERE")
            dp = DataProcesser(data_path, args.train_steps, args.test_steps)
            dp.save_files(data_path)        
    
    epochs = args.num_epochs
    train_dataset = BaseDataset(datasets_names_list, train_steps = args.train_steps, pred_steps = args.test_steps)
    val_dataset = BaseDataset([args.data_dir +"/"+"zara01"], train_steps = args.train_steps, pred_steps = args.test_steps)
    print("Overall dataset: {}".format(len(train_dataset)))
    trainloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers = args.num_workers)    
    valloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers = args.num_workers)    

    hidden_dim = 256
    model = Model(hidden_dim = hidden_dim,train_steps = args.train_steps, predict_steps = args.test_steps, decoder_input_size = hidden_dim, cnn = args.cnn, attention = True)
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr, weight_decay = 1e-5)

    
    train(model,criterion, optimizer, trainloader, valloader, args.num_epochs)
    
    validate(model, valloader)