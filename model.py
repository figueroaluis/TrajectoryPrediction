import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.encoders import CustomCNN
import pdb

class CNN(nn.Module):
    def __init__(self, input_width = 720, input_height = 576):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=3),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(num_features = 96, momentum = 0.8),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2), 
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(num_features = 256, momentum = 0.8),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )    
        inp = 95744 # dim after flatten. can define in terms of ip_width, ip_height 
        self.dense = nn.Sequential(            
            nn.Linear(inp, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, 256),
            nn.ReLU(inplace = True)
        )

    def forward(self, image):
        cnn_feats = self.cnn(image)
        # print(cnn_feats.size())
        feats = cnn_feats.view(-1, 95744)
        # print(feats.size())
        return self.dense(feats)

class SceneGRU(nn.Module):    
    def __init__(self, hidden_dim = 128, train_steps = 8, cnn = 'baseline'):
        super(SceneGRU, self).__init__()
        self.train_steps = train_steps        
        if cnn == 'baseline':
            self.cnn = CNN()
        else:
            self.cnn = CustomCNN() #TODO: change cnn here for resnext
        self.sceneGRU = nn.GRU(input_size = 256, hidden_size = hidden_dim, batch_first = True, dropout = 0.2) #why input = 512, batch_first?
    
    def forward(self, input):
        cnn_features = self.cnn(input)
        cnn_features = cnn_features.unsqueeze(dim=1).repeat(1, self.train_steps, 1) # repeating each feature in batch for 8 steps
        return self.sceneGRU(cnn_features)[1]

class GroupGRU(nn.Module):    
    def __init__(self, hidden_dim = 128, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45):
        super(GroupGRU, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace = True)
        )

        self.groupGRU = nn.GRU(input_size = int(neighborhood_radius / grid_radius) * int(360 / grid_angle), 
                                hidden_size = hidden_dim, batch_first = True, dropout = 0.2) # how will this interact with previous layer, batch_first?


    def forward(self, input):
        features = self.dense(input)
        return self.groupGRU(features)[1]


class PersonGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(PersonGRU, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace = True)
        )

        self.personGRU = nn.GRU(input_size = 64, hidden_size = hidden_dim, batch_first = True, dropout = 0.2)

    def forward(self, input):
        features = self.dense(input)
        return self.personGRU(features)[1]

class Encoder(nn.Module):
    def __init__(self, hidden_dim = 128, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45, train_steps = 8, predict_steps = 12, cnn = 'baseline'):
        super(Encoder, self).__init__()
        self.personModel = PersonGRU(hidden_dim = hidden_dim)
        self.groupModel = GroupGRU(hidden_dim = hidden_dim, neighborhood_radius = neighborhood_radius, grid_radius = grid_radius, grid_angle = grid_angle)
        self.sceneModel = SceneGRU(hidden_dim = hidden_dim, train_steps = train_steps, cnn = cnn)
        self.predict_steps = predict_steps

    def forward(self, images, group_features, person_features):
        output = self.personModel(person_features)
        output += self.groupModel(group_features)
        output += self.sceneModel(images)        
        output = output.permute(1,0,2)
        output = output.repeat(1,self.predict_steps,1)
        return output

class Decoder(nn.Module):
    def __init__(self, input_size = 128, hidden_dim = 128):
        super(Decoder, self).__init__()
        self.decoderGRU = nn.GRU(input_size = input_size, hidden_size = hidden_dim, batch_first = True, dropout = 0.2)
        self.dense = nn.Linear(128, 2)
    
    def forward(self, encoder_output):
        decoder_out, _ = self.decoderGRU(encoder_output)
        return self.dense(decoder_out)

class Model(nn.Module):
    def __init__(self, hidden_dim = 128, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45, train_steps = 8, predict_steps = 12, decoder_input_size = 128, cnn = 'baseline'):
        super(Model, self).__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim, neighborhood_radius=neighborhood_radius, grid_radius=grid_radius, grid_angle=grid_angle, train_steps=train_steps, predict_steps=predict_steps, cnn = cnn)
        self.decoder = Decoder(input_size = decoder_input_size, hidden_dim=hidden_dim)        

    def forward(self, images, group_features, person_features):
        encoder_output = self.encoder(images, group_features, person_features)
        return self.decoder(encoder_output)