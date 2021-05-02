#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from models.resnet import ResNet18_OS8
from models.aspp import ASPP, ASPP_Bottleneck

class CustomDeepLabV3(nn.Module):
    def __init__(self, finetune=True):
        super(CustomDeepLabV3, self).__init__()
        base_model = setup_pretrained_deeplabv3(finetune)
        self.resnet = base_model.resnet
        self.aspp = base_model.aspp

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) 
        output = self.aspp(feature_map)
        
        ### output dimensions should be [batch_size, 256, 60, 80]
        return output

class BaseDeepLabV3(nn.Module):
    def __init__(self):
        super(BaseDeepLabV3, self).__init__()

        self.num_classes = 20
        self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) 
        output = self.aspp(feature_map)
        output = F.upsample(output, size=(h, w), mode="bilinear")

        return output
    
def setup_pretrained_deeplabv3(finetune=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BaseDeepLabV3()
    model.load_state_dict(torch.load("./weights/model_13_2_2_2_epoch_580.pth", 
                                     map_location=torch.device(device)))
    
    if not finetune:
        ### sanity check to freeze all model parameters
        for parameter in model.parameters():
            parameter.requires_grad = False
    
    return model