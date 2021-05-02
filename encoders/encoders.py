#!/usr/bin/env python
# coding: utf-8

from torch import nn
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self, encoder_name = 'resnext', pretrained = False, attention = False, input_width = 720, input_height = 576):
        super(CustomCNN, self).__init__()
        self.attn = attention
        if encoder_name == 'vgg':
            setup_encoder = self._setup_vgg_encoder
            self.inp = 202752
        elif encoder_name == 'resnext':
            setup_encoder = self._setup_resnext_encoder
            self.inp = 847872
        elif encoder_name == 'resnet':
            setup_encoder = self._setup_resnet_encoder
            self.inp = 211968
        self.cnn = setup_encoder(pretrained)
            
        self.dense = nn.Sequential(            
            nn.Linear(self.inp, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, 256),
            nn.ReLU(inplace = True)
        )

    def forward(self, image):
        cnn_feats = self.cnn(image)
        
        if self.attn:
            return cnn_feats
        else:
            feats = cnn_feats.view(-1, self.inp)
            return self.dense(feats)
    
    def _setup_resnet_encoder(self, pretrained=True):
        '''
        Sets up ResNet-18 model.
        '''
        model = models.resnet18(pretrained)
        if pretrained:
            ### sanity check to freeze all model parameters
            for parameter in model.parameters():
                parameter.requires_grad = False

        ### drop the classifier at the end
        encoder = nn.Sequential(*list(model.children())[:8])

        return encoder

    def _setup_resnext_encoder(self, pretrained=True):
        '''
        Sets up ResNeXt-50_32x4d model.
        '''
        model = models.resnext50_32x4d(pretrained)
        # if pretrained:
        #     ### sanity check to freeze all model parameters
        #     for parameter in model.parameters():
        #         parameter.requires_grad = False

        ### drop the classifier at the end
        encoder = nn.Sequential(*list(model.children())[:8])

        return encoder
    
    def _setup_vgg_encoder(self, pretrained=True):
        '''
        Sets up VGG-16 model.
        '''
        model = models.vgg16(pretrained)
        if pretrained:
            ### sanity check to freeze all model parameters
            for parameter in model.parameters():
                parameter.requires_grad = False

        ### drop the classifier at the end
        encoder = nn.Sequential(*list(model.features.children()))

        return encoder