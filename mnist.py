import argparse
import os
import sys
from utils.data import create_dataloader, merge_dataloaders
from tree.tree import Node, grow_tree_from_root
import torch

parser = argparse.ArgumentParser()

#omain config
parser.add_argument('--dataset_path', type=str, default='data',
                        metavar='', help='Path for folder containing the dataset root folder')
parser.add_argument('--logs_path', type=str, default='experiment_logs_mnist',
                        metavar='', help='Folder for saving all logs (replaces previous logs in the folder if any)')
parser.add_argument('--root_node_name', type=str, default='Z',
                        metavar='', help='Name for the root node of the tree')
parser.add_argument('--device', type=int, default=0,
                        metavar='', help='GPU device to be used')
parser.add_argument('--amp_enable', action='store_true', help='Enables automatic mixed precision if available (executes faster on modern GPUs')
parser.set_defaults(amp_enable=False)



#architecture/model parameters
parser.add_argument('--nf_g', type=int, default=128,
                        metavar='', help='Number of feature maps for generator.')
parser.add_argument('--nf_d', type=int, default=128,
                        metavar='', help='Number of feature maps for discriminator/classifier.')                    
parser.add_argument('--kernel_size_g', type=int, default=4,
                        metavar='', help='Size of kernel for generators')
parser.add_argument('--kernel_size_d', type=int, default=5,
                        metavar='', help='Size of kernel for discriminator/classifier')
parser.add_argument('--normalization_d', type=str, default='layer_norm',
                        metavar='', help='Type of normalization layer used for discriminator/classifier')
parser.add_argument('--normalization_g', type=str, default='no_norm',
                        metavar='', help='Type of normalization layer used for generator')
parser.add_argument('--architecture_d', type=str, default='cnn',
                        metavar='', help='Specific architecture choice for for discriminator/classifier')
parser.add_argument('--architecture_g', type=str, default='cnn',
                        metavar='', help='Specific architecture choice for for generator')
parser.add_argument('--img_channels', type=int, default=1,
                        metavar='', help='Number of channels used for intended types of images')
parser.add_argument('--latent_dim', type=int, default=100,
                        metavar='', help="Dimension of generator's latent space")
parser.add_argument('--batch_size_real', type=int, default=100,
                        metavar='', help="Minibatch size for real images")
parser.add_argument('--batch_size_gen', type=int, default=100,
                        metavar='', help="Minibatch size for generated images ")
parser.add_argument('--img_dim', type=int, default=28, 
                        metavar='', help="Image dimensions")
parser.add_argument('--shared_features_across_ref', action='store_true', help='Shares encoder features among parallel refinement groups (inactivated by default)')
parser.set_defaults(shared_features_across_ref=False)

#training parameters
parser.add_argument('--lr_d', type=float, default=0.0001,
                        metavar='', help='Learning rate for discriminator')
parser.add_argument('--lr_c', type=float, default=0.00002,
                        metavar='', help='Learning rate for classifier')
parser.add_argument('--lr_g', type=float, default=0.0002,
                        metavar='', help='Learning rate for generator')
parser.add_argument('--b1', type=float, default=0.5,
                        metavar='', help='Adam optimizer beta 1 parameter')
parser.add_argument('--b2', type=float, default=0.999,
                        metavar='', help='Adam optimizer beta 2 parameter')
parser.add_argument('--noise_start', type=float, default=1.0,
                        metavar='', help='Start image noise intensity linearly decaying throughout each GAN/MGAN training')
parser.add_argument('--epochs_raw_split', type=int, default=100,
                        metavar='', help='Number of epochs for raw split training')
parser.add_argument('--epochs_refinement', type=int, default=100,
                        metavar='', help='Number of epochs for refinement training')
parser.add_argument('--diversity_parameter_g', type=float, default=1.0,
                        metavar='', help="Hyperparameter for weighting generators' classification loss component")
parser.add_argument('--no_refinements', type=int, default=6,
                        metavar='', help='Number of refinements in each split')
parser.add_argument('--no_splits', type=int, default=9,
                        metavar='', help='Number of splits during tree growth')
parser.add_argument('--collapse_check_epoch', type=float, default=40,
                        metavar='', help='Epoch after which to check for generation collapse')
parser.add_argument('--sample_interval', type=int, default=10,
                        metavar='', help='No. of epochs between printring/saving training logs')
parser.add_argument('--min_prob_mass_variation', type=float, default=150,
                        metavar='', help='If the total prob mass variation between two consecutive refinements is less than this number, to save up time, the next refinements are skipped for that node')



args = parser.parse_args()
torch.cuda.set_device(args.device)      
dataloader_train = create_dataloader(dataset='mnist', test=False, batch_size=args.batch_size_real,  path=args.dataset_path)
dataloader_test = create_dataloader(dataset='mnist', test=True, batch_size=args.batch_size_real,  path=args.dataset_path)
dataloader_train = merge_dataloaders(dataloader_train, dataloader_test)
root_node = Node(args.root_node_name, dataloader_train.sampler.weights, args.logs_path)
grow_tree_from_root(root_node, dataloader_train, args)
              