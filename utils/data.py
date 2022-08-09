import os
import math
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        self.targets = dataset.targets
        
    def __getitem__(self, index):
        data, target = self.dataset[index]        
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def merge_dataloaders(dl1, dl2):
    dl1.dataset.data = torch.cat([dl1.dataset.data, dl2.dataset.data])
    dl1.dataset.targets = torch.cat([dl1.dataset.targets, dl2.dataset.targets])
    dl1.sampler.weights = torch.cat([dl1.sampler.weights, dl2.sampler.weights])            
    return dl1     
                        

def create_dataloader(dataset, test = False, batch_size = 100,  path='../data/'):
    
    available_datasets = ['fmnist', 'mnist','sop']
    if dataset not in available_datasets:
        raise ValueError('Dataset {} not available, choose from {}'.format(dataset, available_datasets))

    os.makedirs(path, exist_ok=True)
        
    if dataset == 'fmnist' :
        if test:
            sampler = torch.utils.data.WeightedRandomSampler(weights=[1]*10000, num_samples=batch_size, replacement=True, generator=None)
        else:
            sampler = torch.utils.data.WeightedRandomSampler(weights=[1]*60000, num_samples=batch_size, replacement=True, generator=None)

        dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=not(test), download=True,
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize([.5],[.5],[.5])
                          ])),
            batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)
        dataloader.dataset.classes = ['tshirt', 'pants', 'pullov', 'dress', 'coat','sandal', 'shirt', 'sneak', 'bag', 'ank-bt']
        
    elif dataset =='mnist':
        if test:
            sampler = torch.utils.data.WeightedRandomSampler(weights=[1]*10000, num_samples=batch_size, replacement=True, generator=None)
        else:
            sampler = torch.utils.data.WeightedRandomSampler(weights=[1]*60000, num_samples=batch_size, replacement=True, generator=None)

        dataloader = torch.utils.data.DataLoader(
           datasets.MNIST(path, train=not(test), download=True,
                           transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize([.5],[.5],[.5])
                          ])),
           batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)        
        dataloader.dataset.classes = ['zero', 'one', 'two', 'three', 'four', 'five','six', 'seven', 'eight', 'nine']
        
    elif dataset =='sop':
        sampler = torch.utils.data.WeightedRandomSampler(weights=[1], num_samples=batch_size, replacement=True, generator=None)
        dataloader = torch.utils.data.DataLoader(
           datasets.ImageFolder(os.path.join(path,'Stanford_Online_Products'),                            
                                transform=transforms.Compose([
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize([.5],[.5],[.5])
                          ])),
           batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)   
        dataloader.sampler.weights = torch.Tensor([1]*len(dataloader.dataset))
    return dataloader
