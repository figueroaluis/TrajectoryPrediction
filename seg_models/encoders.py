#!/usr/bin/env python
# coding: utf-8

import torchvision.models as models

def set_up_vgg16(pretrained=False):
    '''
    Sets up VGG-16 model without the maxpool2d layers
    '''
    vgg = models.vgg16(pretrained)
    features = list(vgg.features.children())
    enc1 = nn.Sequential(*features[0:4])
    enc2 = nn.Sequential(*features[5:9])
    enc3 = nn.Sequential(*features[10:16])
    enc4 = nn.Sequential(*features[17:23])
    enc5 = nn.Sequential(*features[24:30])

    return enc1, enc2, enc3, enc4, enc5

def set_up_resnet18(pretrained=False):
    '''
    Setes up ResNet-18 model.
    '''
    resnet = models.resnet18(pretrained)
    features = list(resnet.children())
    enc1 = nn.Sequential(*features[0:4])
    enc2 = nn.Sequential(*features[4])
    enc3 = nn.Sequential(*features[5])
    enc4 = nn.Sequential(*features[6])
    enc5 = nn.Sequential(*features[7])

    return enc1, enc2, enc3, enc4, enc5