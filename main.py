import torch
import numpy as np
import argparse
from data import BaseDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load hyperparams')    
    parser.add_argument('--image_dir', type=str, default="")
    parser.add_argument('--obs_path', type=str, default="")
    parser.add_argument('--train_annotation_path', type=str, default="../DATA/mscoco_train2014_annotations.json")
    parser.add_argument('--test_image_dir', type=str, default="../DATA/val2014/")
    parser.add_argument('--test_question_path', type=str, default="../DATA/OpenEnded_mscoco_val2014_questions.json")    
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_data_loader_workers', type=int, default=4)    
    parser.add_argument('--lr', type=float, default=4e-4)    
    args = parser.parse_args()


    epochs = args.num_epochs
    train_dataset = BaseDataset(args.obs_path, args.pred_path, args.raw_path, args.image_dir)
    trainloader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers = args.num_data_loader_workers)

