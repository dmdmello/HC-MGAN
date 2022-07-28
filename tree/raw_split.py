#basic imports
import argparse
import os
import numpy as np
import math
import shutil
import time
import datetime 
import copy
import sys

#torch imports
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

#plot imports
#import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
#from tabulate import tabulate

#other imports
from tqdm import autonotebook
from sklearn import metrics
#import nbimporter

#custom imports
from utils.soft_cluster import get_local_cluster_table, show, distribution_select
from utils.others import create_gans, sum_dicts, save_log_text, get_log_heading, print_save_log, zero_dict_values

try: 
    from IPython.display import Image
except:
    print('Jupyter image display not available')

class MGANTrainer:
    def __init__(self, 
                 dataloader_cluster_k, #dataloader object
                 mgan_k, #mgan object
                 amp_enable, #bool
                 prior_distribution = 'uniform',
                 node_k = None,
                 feat_extractor = None
                ):
        
        self.dl_k = dataloader_cluster_k
        self.mgan_k = mgan_k
        self.latent_dim = mgan_k.latent_dim
        self.no_c_outputs = mgan_k.clasf.linear_clasf.out_features
        self.amp_scaler = torch.cuda.amp.GradScaler()
        self.amp_autocast = torch.cuda.amp.autocast
        self.amp_enable = amp_enable
        self.cancel_training_flag = False
        self.class_to_idx_dict = dataloader_cluster_k.dataset.class_to_idx.items()
        self.idx_to_class_dict = {v:k for k,v in self.class_to_idx_dict}
        self.classes_names_dict = {v:k for k,v in dataloader_cluster_k.dataset.class_to_idx.items()}
        self.classes_targets_groups = [ dataloader_cluster_k.dataset.class_to_idx[class_name] for class_name in dataloader_cluster_k.dataset.classes]
        self.prior_distribution = prior_distribution
        self.node_k = node_k
        self.feat_extractor = feat_extractor

        self.mgan_k.assign_amp(self.amp_autocast, self.amp_scaler)
        self.mgan_k.enable_amp(self.amp_enable)
        
        CUDA = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
        self.no_batches = int(dataloader_cluster_k.sampler.weights.sum().item())//dataloader_cluster_k.batch_size       

    def train(self, epochs, text_logs_path, raw_split_path, noise_start=0, sample_interval=20, collapse_check_loss=0.001, collapse_check_epoch=0, 
              batch_size_gen=100, raw_split_attempt=0):
        '''Main training loop.

        Args:
        epochs (int): total training epochs          
        text_logs_path (string): .txt file to save textual training logs
        raw_split_path (string): path to raw split folder where logs will be stored
        noise_start (float): Start image noise intensity linearly decaying throughout the training
        sample_interval (int): interval for sample logs printing/saving
        collapse_check_loss (float): threshold discriminator loss for detecting collapsed generators and halting the training
        batch_sige_gen (int): no. of samples per minibatch for generated images
        ref_it (int): no. of iteration of refinement for printing/saving logs
        ref_attempt (int): no. of attempt for a given refinement it. (counts +1 if previous attempt was halted due to generators collapse)   
        '''
        
        self.cancel_training_flag = False
        print("\n\nTraining epochs progress bar (training logs printed/saved every {} epochs):".format(sample_interval))
        for epoch in autonotebook.tqdm(range(1, epochs+1)):
            img_noise_scale = noise_start*(1-epoch/epochs)
            epoch_start = time.time()
                        
            #running losses/acc dictionary
            epoch_metrics_dict = zero_dict_values(copy.copy(self.mgan_k.metrics_dict))
            epoch_metrics_dict = self.train_on_epoch(epoch_metrics_dict, img_noise_scale, batch_size_gen)
            epoch_interval = time.time() - epoch_start

            #logs
            if (epoch % sample_interval) == 0:
                #text_logs_path = self.raw_split_path + "attempt_{}_ep_{}_logs.txt".format(raw_split_attempt, epoch)
                self.view_log_headings(epoch, epochs, epoch_interval, text_logs_path, raw_split_attempt)
                self.view_epoch_losses(epoch_metrics_dict, text_logs_path)
                self.view_gen_imgs(epoch, raw_split_attempt, raw_split_path, text_logs_path)                
                self.verify_collapsed_generators(epoch, text_logs_path, img_noise_scale, collapse_check_loss, collapse_check_epoch)

                #flag for cancelling the training if generators collapse is detected
                if self.cancel_training_flag:
                    break
        
        #prints end of training logs
        print_save_log("\n\n"+get_log_heading("END OF RAW SPLIT TRAINING FOR NODE {}".format(self.node_k.name)), text_logs_path)
        print_save_log("End of training.", text_logs_path) 

        if not(self.cancel_training_flag):

            #gets cluster assignment probabilities for each example with classifier's inference
            clasf_cluster_probs = self.get_clasf_cluster_probs(self.dl_k, self.mgan_k.clasf, img_noise_scale)
            
            # creates children with new clasf cluster probs
            self.node_k.create_children(cluster_probs_left = clasf_cluster_probs[0], cluster_probs_right = clasf_cluster_probs[1])

            #logs with the binary clustering result for current node's raw split
            new_cluster_prob_mass_per_class = self.get_cluster_prob_mass_per_class(self.dl_k, clasf_cluster_probs)
            self.view_new_clasf_clustering(new_cluster_prob_mass_per_class, text_logs_path)    

    
    def train_on_epoch(self, epoch_metrics_dict, img_noise_scale, batch_size_gen=100):     
        mgan_k = self.mgan_k
        for batch_idx in range(self.no_batches):
            #samples real images from groups l and m
            imgs_real_k = self.get_real_images(img_noise_scale)

            #Trains group l components with needed external data/components from group m
            imgs_gen_k = self.get_gen_images(img_noise_scale, batch_size_gen)
            batch_metrics_dict = mgan_k.train_on_batch(imgs_real = imgs_real_k, 
                                                       imgs_gen = imgs_gen_k)
            epoch_metrics_dict = sum_dicts(epoch_metrics_dict, batch_metrics_dict)
             
            #updates amp scaler after training components from groups l and m
            self.amp_scaler.update()

        return epoch_metrics_dict

    def _get_classes_for_monitoring(self, cluster_prob_mass_per_class, min_proportion=0.2):
        clusters_per_class_sum = [np.sum(clusters) 
                                  for clusters in cluster_prob_mass_per_class]
        classes = self.dl_k.dataset.classes
        targets_per_example = self.dl_k.dataset.targets
        examples_per_class_sum = [np.sum(np.array(targets_per_example)==i) 
                                  for i in range(len(classes))]
        classes_for_monitoring = [i for (i, sum_value) in enumerate(clusters_per_class_sum) 
                                       if sum_value > examples_per_class_sum[i] * min_proportion]
        if len(classes_for_monitoring) < 2: 
            print('\nClasses for metrics monitoring were set to {}, ' 
                  'which is too few (2 or more classes required)'.format(classes_for_monitoring))
            print('This means clusters are too small (prob. mass for classes < {}} of original mass at root node).'.format(min_proportion))
            print('Enabling all classes for metrics monitoring.')
            classes_for_monitoring = np.arange(len(classes)).tolist()
            return classes_for_monitoring
        else:
            return classes_for_monitoring

    def get_real_images(self, img_noise_scale):
        '''Gets real images from groups l and m'''

        imgs_real_k = next(iter(self.dl_k))[0].type(self.Tensor)
        imgs_real_k = self._add_noise(imgs_real_k, img_noise_scale)        
        return imgs_real_k
        
    def get_clasf_cluster_probs(self, dataloader, classifier, img_noise_scale=0): 
        '''Performs cluster inference over the entire training set w/ the 1 classifier.'''
        
        dataloader=torch.utils.data.DataLoader(dataloader.dataset, batch_size=100, shuffle=False, drop_last=False)
        
        #empty sublists to accumulate the minibatches of probabilities 
        clasf_cluster_probs = [[] for _ in range(self.no_c_outputs)]
        
        #iterates through the dataset to collect classifiers inference with minibatches
        for (batch_imgs, batch_targets) in dataloader:
            batch_imgs = batch_imgs.cuda()
            batch_imgs = self._add_noise(batch_imgs, img_noise_scale)        
            with torch.no_grad():
                clasf_cluster_probs_batch = torch.exp(classifier(batch_imgs)).transpose(1,0)
                for i in range(self.no_c_outputs):
                    clasf_cluster_probs[i].append(clasf_cluster_probs_batch[i])

        #concatenates results for each batch of the whole data 
        clasf_cluster_probs = np.array([torch.cat(batch).cpu().numpy() for batch in clasf_cluster_probs]) 
                               
        #gets parent node (k) probabilities by summing previous probabilities in l and m
        current_cluster_probs = self.dl_k.sampler.weights.numpy()
        
        #multiplies clasf inference by the current node`s probabilities 
        clasf_cluster_probs[0] *= current_cluster_probs
        clasf_cluster_probs[1] *= current_cluster_probs

        return clasf_cluster_probs.tolist()

    def get_cluster_prob_mass_per_class(self, dataloader, cluster_probs_per_example):
        no_of_clusters = 2
        assert(len(cluster_probs_per_example) == no_of_clusters)
        no_of_classes = len(dataloader.dataset.classes)
        prob_mass_per_class = []
        for i in range(no_of_classes):
            prob_mass_ij = []
            for j in range(no_of_clusters):
                prob_mass_ij.append( ((np.array(dataloader.dataset.targets)==i)*cluster_probs_per_example[j]).sum().item() )
            prob_mass_per_class.append(prob_mass_ij)

        return  np.round(prob_mass_per_class, 2)

    def get_cluster_assignments_per_class(self, dataloader, cluster_probs_per_example): 
        no_of_clusters = 2
        assert(len(cluster_probs_per_example) == no_of_clusters)
        no_of_classes = len(dataloader.dataset.classes)
        cluster_assignments_per_class = []
        for i in range(no_of_classes):
            cluster_counts_ij = []
            for j in range(no_of_clusters):
                cluster_counts_ij.append( ((np.array(dataloader.dataset.targets)==i)*(cluster_probs_per_example[j])>0.5).sum().item() )
            cluster_assignments_per_class.append(cluster_counts_ij)
        return cluster_assignments_per_class

    def view_log_headings(self, epoch, epochs, epoch_interval, text_logs_path, raw_split_attempt=0):
        '''Part 1/4 of training logs'''
        
        log_headings = "[RAW SPLIT NODE %s] [EPOCH %d/%d] [EPOCH TIME INTERVAL: %.2f sec.] [ATTEMPT %d]"%(self.node_k.name, epoch, epochs, epoch_interval, raw_split_attempt)        
        log_headings = get_log_heading(log_headings)
        print_save_log('\n\n' + log_headings, text_logs_path)

    def view_epoch_losses(self, epoch_metrics_dict, text_logs_path):
        '''Part 2/4 of training logs'''

        log_string = 'Mean epoch losses/acc for each component in the MGAN: \n' 
        log_string += str({k:np.round(v/self.no_batches,5) for k,v in epoch_metrics_dict.items()}) + '\n'
        print_save_log(log_string, text_logs_path)

    def view_gen_imgs(self, epoch, raw_split_attempt, raw_split_path, text_logs_path):
        '''Part 3/4 of training logs'''

        imgs_plot = self.get_gen_images(img_noise_scale=0, batch_size=20)
        if self.node_k is not None:
            if raw_split_path is not None:
                img_save_path =  raw_split_path + "attempt_{}_ep_{}.jpg".format(raw_split_attempt, epoch)
                self._plot_img_grid(imgs_plot, img_save_path, self.node_k.name, text_logs_path)

    def verify_collapsed_generators(self, epoch, text_logs_path, img_noise_scale=0, collapse_check_loss=0.01, collapse_check_epoch=50, batch_size=100):
        '''Part 4/4 of training logs'''
        if epoch < collapse_check_epoch:
            print_save_log("\nGenerator collapse will be checked after epoch {}".format(collapse_check_epoch), text_logs_path)
        else:
            imgs_gens_k = self.get_gen_images(img_noise_scale, batch_size)
            losses = self.mgan_k.get_disc_losses_for_gen(imgs_gens_k)
            for loss in losses:
                if loss < collapse_check_loss and epoch>=collapse_check_epoch:
                    log_string = "\nDiscriminator loss for generated images is too low (<{}), indicating generators collapse. The training shall restart.".format(collapse_check_loss)
                    print_save_log(log_string, text_logs_path)
                    self.cancel_training_flag = True
                    break
            if not(self.cancel_training_flag):
                print_save_log("\nGenerator collapse check: no collapse detected, training shall continue.", text_logs_path)
            else: 
                print_save_log("\nGenerator collapse check: collapse detected, restart training.", text_logs_path)

    def view_new_clasf_clustering(self, new_cluster_prob_mass_per_class, text_logs_path):
        '''Prints logs with binary clustering result for current node'''

        #header
        log_headings = "EXHIBITING BINARY CLUSTERING FOR NODE %s OBTAINED WITH CLASSIFIER'S INFERENCE"%(self.node_k.name)     
        log_headings = get_log_heading(log_headings)
        print_save_log("\n\n"+log_headings, text_logs_path)

        #clustering table
        log_text_1 = 'Local binary soft clustering (prob. mass division) for node {}, according to each reference class\n'.format(self.node_k.name)
        print_save_log(log_text_1, text_logs_path)
        table_df = get_local_cluster_table(new_cluster_prob_mass_per_class, self.idx_to_class_dict, node=self.node_k, 
                                           table_name = 'Local soft clusters from binary split')
        pd.set_option("max_colwidth", None)
        pd.set_option('max_columns', None)
        try:
            display(table_df)
        except:
            print(table_df)
        print_save_log(str(table_df), text_logs_path, print_log=False)

    def _add_noise(self, tensor, normal_std_scale):
        if (normal_std_scale > 0):
            return tensor + (tensor*torch.randn_like(tensor)*normal_std_scale)
        else:
            return tensor

    def get_gen_images(self, img_noise_scale, batch_size=100):
        '''Generates imgs from each gan (already concatenated per gan)'''

        latent_dim = self.mgan_k.latent_dim
        z = self.Tensor(distribution_select(self.prior_distribution, (batch_size, latent_dim))).requires_grad_(False)
        imgs_gen  = self.mgan_k.get_gen_images(z, rand_perm=False)
        imgs_gen = self._add_noise(imgs_gen, img_noise_scale)
        return imgs_gen

    def _plot_img_grid(self, imgs_plot, img_save_path, node_name, text_logs_path):
        if imgs_plot.shape[1] == 3 or imgs_plot.shape[1] == 1:
            grid = make_grid(imgs_plot.cpu(), nrow=20, normalize=True)
            if img_save_path is not None:
                save_image(grid, img_save_path)     
                try:
                    print_save_log("\nSample of generated images from raw split MGAN for node {} (each row for each generator' output):".format(node_name), text_logs_path)  
                    print_save_log('(This sample is saved at {})'.format(img_save_path), text_logs_path)   
                    display(Image(filename=img_save_path, width=900))
                except:
                    print_save_log('Jupyter image display not available for plotting sample of generated images', text_logs_path)    
                    #print_save_log("Sample of generated images (each row for each generator' output) saved at {}".format(img_save_path), text_logs_path)   
            else:
                print_save_log("\nNo image save path defined, can't save sample of generated images", text_logs_path)
        else:
            print_save_log("\nCan't plot/save imgs with shape {}".format(imgs_plot.shape), text_logs_path)   

def raw_split(args, dataloader_cluster_k, node_k, epochs, noise_start, sample_interval=10, collapse_check_loss=0.001):
    
    restart_training = True
    raw_split_attempt = 1
    max_attempts = 4

    while restart_training:
        
        #configure log saving paths
        raw_split_path = os.path.join(node_k.node_path, "raw_split/")
        os.makedirs(raw_split_path, exist_ok=True)
        text_logs_path = raw_split_path + "attempt_{}_training_logs.txt".format(raw_split_attempt)
        save_log_text('', text_logs_path, open_mode='w')

        #print main log headings
        log_headings = get_log_heading("RAW SPLIT OF NODE {} (ATTEMPT {}) ".format(node_k.name, raw_split_attempt), spacing=2)
        print_save_log('\n\n\n' + log_headings, text_logs_path)        

        #print parameters
        log_headings = get_log_heading("TRAINING PARAMETERS")
        print_save_log(log_headings, text_logs_path)
        print_save_log("Training Arguments: ", text_logs_path)
        print_save_log(vars(args), text_logs_path)
        print_save_log("Training using device : {}".format(args.device), text_logs_path)
        print_save_log("Training logs save path: {}".format(text_logs_path), text_logs_path)
        print_save_log("Limit of Training Attempts: {}".format(max_attempts), text_logs_path)

        #create MGAN models
        [mgan_k] = create_gans(args, no_gans=1, no_g_paths=2)

        #print models' architecture
        log_headings = get_log_heading("MODELS ARCHITETURE")
        print_save_log('\n\n' + log_headings, text_logs_path)        
        print_save_log("Discriminator Architecture:", text_logs_path)
        print_save_log(mgan_k.disc, text_logs_path)
        print_save_log("\nGernerator Architecture:", text_logs_path)
        print_save_log(mgan_k.gen_set.paths[0], text_logs_path)      
        print_save_log("\nClassifier Architecture:", text_logs_path)
        print_save_log(mgan_k.clasf, text_logs_path)

        #create trainer object
        trainer = MGANTrainer(dataloader_cluster_k, 
                              mgan_k, 
                              amp_enable=args.amp_enable,
                              prior_distribution = "uniform",
                              node_k = node_k
                              )

        #train      
        trainer.train(epochs = epochs, 
                      text_logs_path = text_logs_path,
                      raw_split_path = raw_split_path,
                      noise_start=noise_start, 
                      sample_interval=sample_interval, 
                      collapse_check_loss =collapse_check_loss, 
                      collapse_check_epoch = args.collapse_check_epoch,
                      raw_split_attempt = raw_split_attempt,
                      )

        #flag for restarting the training if generation collapse is detected
        if trainer.cancel_training_flag == False:
            restart_training = False
        else:
            raw_split_attempt += 1

        if raw_split_attempt>max_attempts:
            max_attempt_log_headings = get_log_heading("LIMIT OF {} FAILED ATTEMPTS REACHED".format(max_attempts))
            max_attempt_log = "The training for the raw split of node {} reached the limit of {} failed attempts due generation collapse.".format(node_k, max_attempts)
            max_attempt_log += " Please, select more stable tunnings for the models so that the generation stops collapsing."
            print_save_log("\n\n" + max_attempt_log_headings, text_logs_path)
            print_save_log(max_attempt_log, text_logs_path)
            sys.exit(max_attempt_log)

    return trainer








