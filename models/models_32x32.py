import argparse
import os
from torch.autograd import Variable
import torch.nn as nn
import torch
from models.utils import verify_string_args, linear_block, Reshape, convT_block, conv_block

class Generator(nn.Module):
    def __init__(self, 
                 architecture = 'cnn', 
                 nf=128, 
                 kernel_size=4, 
                 latent_dim = 100, 
                 nc = 3,
                 print_shapes=False,
                 norm = 'no_norm'
                ):
       
        super(Generator, self).__init__()
        print_shapes = False        
        architecture_list = ['cnn', 'cnn_short', 'cnn_long']
        normalization_list = ['no_norm']
        
        verify_string_args(architecture, architecture_list)
        verify_string_args(norm, normalization_list)        
        
        self.img_size = 32
        self.architecture = architecture
        self.nf = nf
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.nc = nc
        self.norm = norm
        
        #print('Generator normalization is ', self.norm)

        gen_layers = []
        
        if architecture == 'cnn' or architecture == 'cnn_short':
            first_map_shape = 8
            gen_layers += linear_block(self.latent_dim, nf*2*first_map_shape*first_map_shape, norm='no_norm', act=nn.ReLU(True))
            gen_layers += Reshape(-1, nf*2, first_map_shape, first_map_shape),
            gen_layers += convT_block(nf*2, nf, stride=2, padding=1, norm=self.norm, act=nn.ReLU(True)) 
            gen_layers += convT_block(nf, nc, stride=2, padding=1, norm='no_norm', act=nn.Tanh())  

        elif (architecture == 'cnn_long'):
            first_map_shape = 3
            gen_layers += linear_block(self.latent_dim, nf*4*first_map_shape*first_map_shape, norm='no_norm', act=nn.ReLU(True))
            gen_layers += Reshape(-1, nf*4, first_map_shape, first_map_shape),
            gen_layers += convT_block(nf*4, nf*2, stride=2, padding=1, norm=self.norm, act=nn.ReLU(True)) 
            gen_layers += convT_block(nf*2, nf, stride=2, padding=0, norm=self.norm, act=nn.ReLU(True))  
            gen_layers += convT_block(nf, nc, stride=2, padding=1, norm='no_norm',  act=nn.Tanh())  
            
        else:
            raise ValueError('Architecture {} not implemented!'.format(architecture))
            
        self.generate = nn.Sequential(*gen_layers)
        
        if print_shapes:
            input_tensor = torch.zeros(100,self.latent_dim)
            output = input_tensor
            print("\nGenerator ConvT Shapes:\n")
            for i, ly in enumerate(self.generate):
                output = self.generate[i](output)
                if (type(ly) == torch.nn.modules.conv.ConvTranspose2d):
                    print('layer: {}'.format(i))
                    print(ly)
                    print('output shape: {}'.format(output.shape))
        
    def forward(self, z):
        
        img = self.generate(z)
        
        if self.architecture == 'mlp':
            img = img.view(-1,self.nc, self.img_size, self.img_size)
            
        return img


class EncoderLayers(nn.Module):
    def __init__(self, 
                 architecture='cnn',
                 nf=128, 
                 kernel_size=5, 
                 norm = 'no_norm',
                 nc = 3,
                 print_shapes=True
                ):

        super(EncoderLayers, self).__init__()
        print_shapes = False
            
        architecture_list = ['cnn', 'cnn_short', 'cnn_long']
        normalization_list = ['layer_norm', 'spectral_norm', 'no_norm']
        verify_string_args(architecture, architecture_list)
        verify_string_args(norm, normalization_list)        
        
        self.img_size = 32
        self.architecture = architecture
        self.nf = nf
        self.kernel_size = kernel_size
        self.norm = norm
        self.nc = nc
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        #print('Normalization for conv layers is {}'.format(norm))

        encoder_layers = []

        if (architecture == 'cnn' or architecture == 'cnn_short'):
            encoder_layers += conv_block(nc, nf, fmap_shape=[16, 16], norm=self.norm, act=self.leaky_relu, kernel_size=self.kernel_size)
            encoder_layers += conv_block(nf, nf * 2, fmap_shape=[8, 8], norm=self.norm, act=self.leaky_relu, kernel_size=self.kernel_size) 
            encoder_layers += conv_block(nf * 2, nf * 4, fmap_shape=[4,4], norm=self.norm, act=self.leaky_relu, kernel_size=self.kernel_size) 
        
        else:
            print('Architecture {} not implemented!'.format(architecture))
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        if print_shapes: 
            print("\nConv Features Shapes\n")        
            
        input_tensor = torch.zeros(100, self.nc, self.img_size, self.img_size)
        output=input_tensor
        if architecture == 'mlp':
            output = input_tensor.view(100,-1)

        for i, ly in enumerate(self.encoder_layers):
            output = self.encoder_layers[i](output)
            if (type(ly) == torch.nn.modules.conv.Conv2d and print_shapes):
                print('layer: {}'.format(i))
                print(ly)
                print('output shape: {}'.format(output.shape))

        self.total_units = output.view(input_tensor.shape[0], -1).shape[-1]
        
    def forward(self, img):
        img_input_dim = img.shape[-1]
        if img_input_dim!=self.img_size:
            raise Exception("This discriminator/classifier assumes image inputs with {} resolution and an input with {} resolution was received. Please choose a compatible model or data.".format(self.img_size, img_input_dim))
        if self.architecture == 'mlp': 
            img = img.view(img.shape[0],-1)
        return self.encoder_layers(img)
 