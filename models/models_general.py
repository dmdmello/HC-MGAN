import torch.nn as nn
import torch.nn.functional as F
import torch

class GeneratorSet(nn.Module):
    def __init__(self, *gens):
        
        super(GeneratorSet, self).__init__()
        modules = nn.ModuleList()
        for gen in gens:
            modules.append(gen)
        self.paths = modules

    def forward(self, z, rand_perm=False):
        
        img = []
        for path in self.paths:
            img.append(path(z))
        img = torch.cat(img, dim=0)
        if rand_perm:
            img = img[torch.randperm(img.shape[0])]
            
        return img

class Classifier(nn.Module):
    def __init__(self, 
                 feature_layers, 
                 no_c_outputs = 2, 
                 dropout = 0
                ):
        
        super(Classifier, self).__init__()
        
        self.feature_layers = feature_layers
        self.no_c_outputs = no_c_outputs
        total_units = feature_layers.total_units
        self.linear_clasf = nn.Linear(total_units, no_c_outputs)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, feature_input = False):
        
        if feature_input:
            conv_features = input_tensor
        else:
            conv_features = (self.feature_layers(input_tensor))
        
        conv_features = conv_features.view(conv_features.shape[0], -1)
        classification = self.dropout(conv_features)
        classification = self.linear_clasf(classification)
        classification = self.log_softmax(classification)
        
        return classification

class Discriminator(nn.Module):
    def __init__(self, 
                 feature_layers
                ):
        
        super(Discriminator, self).__init__()
        self.feature_layers = feature_layers
        total_units = feature_layers.total_units
        self.linear_disc  = nn.Linear(total_units, 1)
        
    def forward(self, input_tensor, feature_input = False):
        
        if feature_input:
            conv_features = input_tensor
        else:
            conv_features = (self.feature_layers(input_tensor))
        conv_features = conv_features.view(conv_features.shape[0], -1)
        validity = self.linear_disc(conv_features)

        return validity

