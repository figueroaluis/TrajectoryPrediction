import torch
import torch.nn as nn
import numpy as np
import argparse
from data import BaseDataset
from torch.utils.data import DataLoader
import os
from utils import DataProcesser


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load hyperparams')    
    parser.add_argument('--image_dir', type=str, default="")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)    
    parser.add_argument('--lr', type=float, default=4e-4)    
    args = parser.parse_args()

    if not os.path.exists(args.data_dir + "/obs.npy"):                    
        dp = DataProcesser(8, 12)
        dp.save_files(args.data_dir)

    epochs = args.num_epochs
    train_dataset = BaseDataset(args.data_dir + "/obs.npy", args.data_dir + "/pred.npy", args.data_dir + "/pixel_pos.csv", args.image_dir)
    trainloader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers = args.num_workers)    

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(weight_decay = 1e-5)