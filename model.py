import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_width = 720, input_height = 576):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=3),
            nn.MaxPool2D(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(num_features = 96, momentum = 0.8),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2), 
            nn.MaxPool2D(kernel_size = 3, stride = 2),
            nn.BatchNorm2d(num_features = 256, momentum = 0.8),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size = 3, stride = 2)
        )    
        inp = 91392 # dim after flatten. can define in terms of ip_width, ip_height 
        self.dense = nn.Sequential(            
            nn.Linear(inp, 512),
            nn.ReLU(inplace = True),
            nn.Linear(inp, 256),
            nn.ReLU(inplace = True)
        )

    def forward(self, image):
        cnn_feats = self.cnn(image)
        feats = torch.flatten(cnn_feats)
        return self.dense(feats)

class SceneGRU(nn.Module):    
    def __init__(self, hidden_dim = 128):
        super(SceneGRU, self).__init__()
        self.cnn = CNN()
        self.sceneGRU = nn.GRU(input_size = 256, hidden_size = hidden_dim, batch_first = True, dropout = 0.2) #why input = 512, batch_first?
    
    def forward(self, input):
        cnn_features = self.cnn(input)
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

        self.personGRU = nn.GRU(input_size = 64, hidden_size = hidden_dim, batch_first = True, dropout = 0.2) # how will this interact with previous layer, batch_first?


    def forward(self, input):
        features = self.dense(input)
        return self.personGRU(features)[1]

class Encoder(nn.Module):
    def __init__(self, hidden_dim = 128, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45):
        super(Encoder, self).__init__()
        self.personModel = PersonGRU(hidden_dim)
        self.groupModel = GroupGRU(hidden_dim, neighborhood_radius, grid_radius, grid_angle)
        self.sceneModel = SceneGRU(hidden_dim)

    def forward(self, images, group_features, person_features):
        output = self.personModel(person_features)
        output += self.groupModel(group_features)
        output += self.sceneModel(images)
        output = output.permute(1,0,2)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoderGRU = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True, dropout = 0.2)
        self.dense = nn.Linear(128, 2)
    
    def forward(self, encoder_feats):
        decoder_out, _ = self.decoder(encoder_feats)
        return self.dense(decoder_out)
