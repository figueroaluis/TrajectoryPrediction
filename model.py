import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.encoders import CustomCNN
import pdb

class CNN(nn.Module):
    def __init__(self, input_width = 720, input_height = 576, hidden_dim = 128, attention = False):
        super(CNN, self).__init__()
        self.attn = attention
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
            nn.Linear(inp, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace = True)
        )

    def forward(self, image):
        cnn_feats = self.cnn(image)
        # print(cnn_feats.size())

        if not self.attn:
            feats = cnn_feats.view(-1, 95744)
            # print(feats.size())
            return self.dense(feats)
        else:
            return cnn_feats


class SceneGRU(nn.Module):    
    def __init__(self, hidden_dim = 128, train_steps = 8, cnn = 'baseline', attention = False):
        super(SceneGRU, self).__init__()
        self.train_steps = train_steps     
        self.attn = attention  
        self.name = cnn 
        if cnn == 'baseline':
            self.cnn = CNN(hidden_dim=hidden_dim, attention = attention)
        else:
            self.cnn = CustomCNN(attention=attention) #TODO: change cnn here for resnext
        self.sceneGRU = nn.GRU(input_size = 256, hidden_size = hidden_dim, batch_first = True, dropout = 0.2) #why input = hidden_dim, batch_first?
        
        self.attnCNN = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )   

    def forward(self, input):
        cnn_features = self.cnn(input) # B x 256
        if self.attn:
            op = cnn_features
            if self.name != "baseline":
                op = self.attnCNN(cnn_features)
            op = op.view(-1, 256, op.size(2) * op.size(3))
            return op
        else:
            cnn_features = cnn_features.unsqueeze(dim=1).repeat(1, self.train_steps, 1) # repeating each feature in batch for 8 steps
            return self.sceneGRU(cnn_features)[1] 

class GroupGRU(nn.Module):    
    def __init__(self, hidden_dim = 128, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45, attention = False):
        super(GroupGRU, self).__init__()
        self.attn = attention 
        self.dense = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace = True)
        )

        self.groupGRU = nn.GRU(input_size = int(neighborhood_radius / grid_radius) * int(360 / grid_angle), 
                                hidden_size = hidden_dim, batch_first = True, dropout = 0.2) # how will this interact with previous layer, batch_first?


    def forward(self, input):
        features = self.dense(input)
        if self.attn:
            return self.groupGRU(features)[0]
        else:
            return self.groupGRU(features)[1]


class PersonGRU(nn.Module):
    def __init__(self, hidden_dim, attention = False):
        super(PersonGRU, self).__init__()
        self.attn = attention 
        self.dense = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace = True)
        )

        self.personGRU = nn.GRU(input_size = 64, hidden_size = hidden_dim, batch_first = True, dropout = 0.2)

    def forward(self, input):
        features = self.dense(input)
        if self.attn:
            return self.personGRU(features)[0] # if dim 0 then B x 5 x 128
        else:
            return self.personGRU(features)[1]

class Encoder(nn.Module):
    def __init__(self, hidden_dim = 256, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45, train_steps = 8, predict_steps = 12, cnn = 'baseline', attention = False):
        super(Encoder, self).__init__()
        self.attn = attention 
        self.personModel = PersonGRU(hidden_dim = hidden_dim, attention=attention)
        self.groupModel = GroupGRU(hidden_dim = hidden_dim, neighborhood_radius = neighborhood_radius, grid_radius = grid_radius, grid_angle = grid_angle, attention=attention)
        self.sceneModel = SceneGRU(hidden_dim = hidden_dim, train_steps = train_steps, cnn = cnn, attention=attention)
        self.predict_steps = predict_steps

        ########################## TEMP##################################
                #Attention
        self.w_x_img = nn.Linear(hidden_dim, hidden_dim) # done
        self.w_x_txt = nn.Linear(hidden_dim,hidden_dim) # done
        self.w_x_group = nn.Linear(hidden_dim,hidden_dim) 

        self.a_x_img = nn.Linear(hidden_dim,1)  #done
        self.a_x_group = nn.Linear(hidden_dim,1)
        
        self.a_x_txt = nn.Linear(hidden_dim,1)  #done
        self.relu = nn.Tanh()                   #done
        self.softmax = nn.Softmax(dim=1) ## done


        #encoding for prediction
        self.person_level = nn.Linear(hidden_dim,hidden_dim)
        self.group_level = nn.Linear(hidden_dim*2,hidden_dim)


        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)
        self.d3 = nn.Dropout(0.5)
        self.d4 = nn.Dropout(0.5)
        self.d5 = nn.Dropout(0.5)

        ####################################################################
    
    def compute_H(self, img, txt, prev_attention_vec, iter):

        if iter==0:
            H = self.d1(self.relu(self.w_x_txt(txt)))
            return H
        
        if iter==1:
            g = self.w_x_txt(prev_attention_vec).unsqueeze(1)   
            H = self.d2(self.relu(self.w_x_img(img) + g))
            return H
            
        if iter==2:
            g = self.w_x_img(prev_attention_vec).unsqueeze(1)
            H =  self.d3(self.relu(self.w_x_txt(txt) + g))
            return H

    def compute_H2(self,txt, group, prev_attention_vec, iter):

        if iter==0:
            H = self.d1(self.relu(self.w_x_txt(txt)))
            return H
        
        if iter==1:
            g = self.w_x_txt(prev_attention_vec).unsqueeze(1)   
            H = self.d2(self.relu(self.w_x_group(group) + g))
            return H
            
        if iter==2:
            g = self.w_x_group(prev_attention_vec).unsqueeze(1)
            H =  self.d3(self.relu(self.w_x_txt(txt) + g))
            return H


    def forward(self, images, group_features, person_features):

        if not self.attn:
            output = self.personModel(person_features)
            output += self.groupModel(group_features)
            output += self.sceneModel(images)        
            output = output.permute(1,0,2)
            output = output.repeat(1,self.predict_steps,1)
            return output
        else:
            person_embedding = self.personModel(person_features)
            group_embedding = self.groupModel(group_features)
            #person level
            image_embedding = self.sceneModel(images).permute(0,2,1)
            person_H = self.compute_H(image_embedding,person_embedding,None,0) # Bx26xhidden_dim
            person_attention_inter = torch.sum(person_embedding * self.softmax(self.a_x_txt(person_H)),1)# Bx26xhidden_dim * Bx26x1 -> Bxhidden_dim
            person_H = self.compute_H(image_embedding,person_embedding,person_attention_inter,1) 
            person_attention_img = torch.sum(image_embedding * self.softmax(self.a_x_img(person_H)),1)
            person_H = self.compute_H(image_embedding, person_embedding, person_attention_img,2)
            person_attention_inter = torch.sum(person_embedding * self.softmax(self.a_x_txt(person_H)),1)

            person_attention = self.d4(self.relu(self.person_level(person_attention_inter + person_attention_img)))


            #group level
            group_H = self.compute_H2(person_embedding, group_embedding,None,0)
            group_attention_inter = torch.sum(person_embedding * self.softmax(self.a_x_txt(group_H)),1)
            group_H = self.compute_H2(person_embedding, group_embedding,group_attention_inter,1)
            group_attention_person = torch.sum(group_embedding * self.softmax(self.a_x_group(group_H)),1) ##person
            group_H = self.compute_H2(person_embedding, group_embedding,group_attention_person,2)
            group_attention_inter = torch.sum(person_embedding * self.softmax(self.a_x_txt(group_H)),1)       


            group_attention_1 = torch.cat([group_attention_person + group_attention_inter, person_attention],-1)
            group_attention = self.d5(self.relu(self.group_level(group_attention_1))).unsqueeze(1)
            output = group_attention.repeat(1,self.predict_steps,1)
            return output

class Decoder(nn.Module):
    def __init__(self, input_size = 128, hidden_dim = 128):
        super(Decoder, self).__init__()
        self.decoderGRU = nn.GRU(input_size = input_size, hidden_size = hidden_dim, batch_first = True, dropout = 0.2)
        self.dense = nn.Linear(hidden_dim, 2)
    
    def forward(self, encoder_output):
        decoder_out, _ = self.decoderGRU(encoder_output)
        return self.dense(decoder_out)

class Model(nn.Module):
    def __init__(self, hidden_dim = 128, neighborhood_radius = 32, grid_radius = 4, grid_angle = 45, train_steps = 8, predict_steps = 12, decoder_input_size = 128, cnn = 'baseline', attention = False):
        super(Model, self).__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim, neighborhood_radius=neighborhood_radius, grid_radius=grid_radius, grid_angle=grid_angle, train_steps=train_steps, predict_steps=predict_steps, cnn = cnn, attention=attention)
        self.decoder = Decoder(input_size = decoder_input_size, hidden_dim=hidden_dim)        

    def forward(self, images, group_features, person_features):
        encoder_output = self.encoder(images, group_features, person_features)
        return self.decoder(encoder_output)