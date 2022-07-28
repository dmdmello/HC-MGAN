from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

def get_seq_model_shapes(seq_model, input_shape, seq_model_name = 'seq_model'):
    input_tensor = torch.zeros(*input_shape)
    output = input_tensor
    print("\n{} Layers:\n".format(seq_model_name))
    
    for i, ly in enumerate(seq_model):
        output = seq_model[i](output)
        print('Layer Block {}: {}, out shape: {}'.format(i, ly, output.shape))
    return output

def verify_string_args(string_arg, string_args_list):
    if string_arg not in string_args_list:
        raise ValueError("Argument '{}' not available in {}".format(string_arg, string_args_list))

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def convT_block(nf_in, nf_out, stride = 2, padding = 1,norm='no_norm', act=None, kernel_size=4):
    block = [nn.ConvTranspose2d(nf_in, nf_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True )]
    if act is not None:
        block.append(act)
    return block

def conv_block(nf_in, nf_out, stride = 2, padding = 2, fmap_shape=[10,10], norm=None, act=None, kernel_size=5):
    block = [nn.Conv2d(nf_in, nf_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True )]
    if norm == 'layer_norm':
        block.append(nn.LayerNorm([nf_out]+fmap_shape))
    elif norm == 'spectral_norm':
        block[-1] = torch.nn.utils.spectral_norm(block[-1])
    if act is not None:
        block.append(act)
    #block.append(nn.LeakyReLU(0.2, inplace=True))
    #block.append(GaussianNoise(normal_std_scale=0.7))
    return block


def linear_block(nf_in, nf_out, norm='no_norm',  act=None):
    block = [nn.Linear(nf_in, nf_out)]
    if norm == 'layer_norm':
        block.append(nn.LayerNorm([nf_out]))
    elif norm == 'spectral_norm':
        block[-1] = torch.nn.utils.spectral_norm(block[-1])
    if act is not None:
        block.append(act)
    return block



