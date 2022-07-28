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
import pandas as pd
import matplotlib.pyplot as plt
#from tabulate import tabulate

#other imports
from tqdm import autonotebook
from sklearn import metrics

#custom imports
from utils.soft_cluster import get_classification_table_variation
from utils.soft_cluster import show, distribution_select
from utils.others import create_gans, sum_dicts, zero_dict_values, save_log_text, get_log_heading, get_bold_string, print_save_log

try: 
    from IPython.display import Image
except:
    print('Jupyter image display not available')

    
class GANGroupsTrainer:
    def __init__(self, 
                 dataloader_l, #dataloader object
                 dataloader_m, #dataloader object
                 gan_l, #gan object
                 gan_m, #gan object
                 amp_enable, #bool
                 prior_distribution = 'uniform',
                 node_k = None
                ):
        
        self.dl_l = dataloader_l
        self.dl_m = dataloader_m
        self.gan_l = gan_l
        self.gan_m = gan_m
        self.no_c_outputs = 2
        self.amp_scaler = torch.cuda.amp.GradScaler()
        self.amp_autocast = torch.cuda.amp.autocast
        self.amp_enable = amp_enable
        self.cancel_training_flag = False
        self.class_to_idx_dict = dataloader_l.dataset.class_to_idx.items()
        self.idx_to_class_dict = {v:k for k,v in self.class_to_idx_dict}
        self.prior_distribution = prior_distribution
        self.node_k = node_k
        self.classifiers = [self.gan_l.clasf, self.gan_m.clasf]
        self.refresh_clustering_attributes()
        self.Tensor = torch.cuda.FloatTensor
        self.node_l = self.node_k.child_left
        self.node_m = self.node_k.child_right
        batch_size = dataloader_l.batch_size   
        self.no_batches = int(dataloader_l.sampler.weights.sum().item() + dataloader_m.sampler.weights.sum().item())//(batch_size*2)    

        self.gan_l.assign_amp(self.amp_autocast, self.amp_scaler)
        self.gan_l.enable_amp(self.amp_enable)
        self.gan_m.assign_amp(self.amp_autocast, self.amp_scaler)
        self.gan_m.enable_amp(self.amp_enable)
    
    def refresh_cluster_probs_per_example(self):
        self.cluster_probs_per_example = [self.dl_l.sampler.weights.numpy(), 
                                          self.dl_m.sampler.weights.numpy()]
        
    def refresh_cluster_prob_mass_per_class(self):   
        self.cluster_prob_mass_per_class = self.get_cluster_prob_mass_per_class(
            self.dl_l, self.cluster_probs_per_example)
        
    def refresh_cluster_assignments_per_class(self):
        self.cluster_assignments_per_class = self.get_cluster_assignments_per_class(
            self.dl_l, self.cluster_probs_per_example)
        
    def refresh_classes_for_monitoring(self):
        clusters_per_class_sum = [np.sum(clusters) 
                                  for clusters in self.cluster_prob_mass_per_class]
        classes = self.dl_l.dataset.classes
        targets_per_example = self.dl_l.dataset.targets
        examples_per_class_sum = [np.sum(np.array(targets_per_example)==i) 
                                  for i in range(len(classes))]
        self.classes_for_monitoring = [i for (i, sum_value) in enumerate(clusters_per_class_sum) 
                                       if sum_value > examples_per_class_sum[i] * 0.2]
        if len(self.classes_for_monitoring) < 2: 
            print('Classes for metrics monitoring were set to {}, ' 
                  'which is too few (2 or more classes required)'.format(
                self.classes_for_monitoring))
            print('This means clusters are too small '
                  '(prob. mass for classes < 20% of original mass at root node).')
            print('Enabling all classes for metrics monitoring.')
            self.classes_for_monitoring = np.arange(len(classes)).tolist()
        
    def refresh_clustering_attributes(self):
        self.refresh_cluster_probs_per_example()
        self.refresh_cluster_prob_mass_per_class()
        self.refresh_cluster_assignments_per_class()
        self.refresh_classes_for_monitoring()
        
    def train(self, epochs, text_logs_path, refinement_path, noise_start=0, sample_interval=20, 
              collapse_check_loss=0.001, collapse_check_epoch=0, batch_size_gen=100, ref_it=0, ref_attempt=1, no_refinements=0):
        '''Main training loop.

        Args:
        epochs (int): total training epochs          
        text_logs_path (string): .txt file to save textual training logs
        refinement_path (string): path to refinement folder where logs will be stored
        noise_start (float): Start image noise intensity linearly decaying throughout the training
        sample_interval (int): interval for sample logs printing/saving
        collapse_check_loss (float): threshold discriminator loss for detecting collapsed generators and halting the training
        batch_sige_gen (int): number of samples per minibatch for generated images
        ref_it (int): no. of iteration of refinement for printing/saving logs
        ref_attempt (int): no. of attempt for a given refinement it. (counts +1 if previous attempt was halted due to generators collapse)   
        '''

        self.refresh_clustering_attributes()
        self.cancel_training_flag = False
        print("\n\nTraining epochs progress bar (training logs printed/saved every {} epochs):".format(sample_interval))
        for epoch in autonotebook.tqdm(range(1, epochs+1)):
            img_noise_scale = noise_start*(1-epoch/epochs)
            epoch_start = time.time()
            
            #running losses/acc dictionary
            epoch_metrics_dict_l = zero_dict_values(copy.copy(self.gan_l.metrics_dict))
            epoch_metrics_dict_m = zero_dict_values(copy.copy(self.gan_m.metrics_dict))
            dicts = self.train_on_epoch(epoch_metrics_dict_l, epoch_metrics_dict_m, img_noise_scale, batch_size_gen)
            epoch_metrics_dict_l, epoch_metrics_dict_m = dicts
            epoch_interval = time.time() - epoch_start
            
            #logs
            if (epoch % sample_interval) == 0:
                self.view_log_headings(epoch, epochs, epoch_interval, text_logs_path, ref_it=ref_it, ref_attempt=ref_attempt)
                self.view_epoch_losses(epoch_metrics_dict_l, epoch_metrics_dict_m, text_logs_path)
                self.view_gen_imgs(epoch, ref_attempt, refinement_path, text_logs_path)                
                self.verify_collapsed_generators(epoch, text_logs_path, img_noise_scale, collapse_check_loss, collapse_check_epoch=collapse_check_epoch)

                #flag for cancelling training if generators collapses is detected
                if self.cancel_training_flag:
                    break

        #prints end of training logs
        end_of_training_logs = "END OF REFINEMENT TRAINING FOR NODE {} SPLIT".format(self.node_k.name)
        print_save_log("\n\n"+get_log_heading(end_of_training_logs), text_logs_path)
        print_save_log("End of training.", text_logs_path) 


        if not(self.cancel_training_flag):
            #gets cluster assignment probabilities for each example, avaraging the 2 classifiers results
            clasf_cluster_probs = self.get_clasf_cluster_probs(self.dl_l, self.classifiers, img_noise_scale)

            # updates children with new refined clasf cluster probs
            self.node_k.child_left.add_cluster_probs(torch.Tensor(clasf_cluster_probs[0]))
            self.node_k.child_left.status = "{}/{} refinements".format(ref_it, no_refinements)
            self.node_k.child_right.add_cluster_probs(torch.Tensor(clasf_cluster_probs[1]))
            self.node_k.child_right.status = "{}/{} refinements".format(ref_it, no_refinements)

            #end of training logs with refined binary clustering for current node
            new_cluster_prob_mass_per_class = self.get_cluster_prob_mass_per_class(self.dl_l, clasf_cluster_probs)
            self.view_new_clasf_clustering(new_cluster_prob_mass_per_class, ref_it, text_logs_path)


    def train_on_epoch(self, epoch_metrics_dict_l, epoch_metrics_dict_m, img_noise_scale, batch_size_gen=100):     
        gan_l = self.gan_l
        gan_m = self.gan_m

        for batch_idx in range(self.no_batches):

            #samples real images from groups l and m
            imgs_real_l, imgs_real_m = self.get_real_images(img_noise_scale)

            #Trains group l components with needed external data/components from group m
            imgs_gen_l, imgs_gen_m = self.get_gen_images(img_noise_scale, batch_size_gen)
            batch_metrics_dict_l = gan_l.train_on_batch_refinement(imgs_real = imgs_real_l, 
                                                                   imgs_gen_internal = imgs_gen_l, 
                                                                   imgs_gen_external = imgs_gen_m,
                                                                   clasf_external = gan_m.clasf)
            epoch_metrics_dict_l = sum_dicts(epoch_metrics_dict_l, batch_metrics_dict_l)
            
            #Trains group m components with needed external data/components from group l
            imgs_gen_l, imgs_gen_m = self.get_gen_images(img_noise_scale, batch_size_gen)
            batch_metrics_dict_m = gan_m.train_on_batch_refinement(imgs_real = imgs_real_m, 
                                                                   imgs_gen_internal = imgs_gen_m, 
                                                                   imgs_gen_external = imgs_gen_l,
                                                                   clasf_external = gan_l.clasf)
            epoch_metrics_dict_m = sum_dicts(epoch_metrics_dict_m, batch_metrics_dict_m)
            
            #updates amp scaler after training components from groups l and m
            self.amp_scaler.update()
        return epoch_metrics_dict_l, epoch_metrics_dict_m
        
    def get_real_images(self, img_noise_scale):
        '''Gets real images from groups l and m'''

        imgs_real_l = next(iter(self.dl_l))[0].type(self.Tensor)
        imgs_real_l = self._add_noise(imgs_real_l, img_noise_scale)
        imgs_real_m = next(iter(self.dl_m))[0].type(self.Tensor)
        imgs_real_m = self._add_noise(imgs_real_m, img_noise_scale)        
        return imgs_real_l, imgs_real_m

    def get_gen_images(self, img_noise_scale, batch_size=100):
        '''Generates imgs from each gan (already concatenated per gan)'''

        latent_dim = self.gan_l.latent_dim
        z = self.Tensor(distribution_select(self.prior_distribution, (batch_size, latent_dim))).requires_grad_(False)
        imgs_gen_l  = self.gan_l.get_gen_images(z, rand_perm=True)
        imgs_gen_l = self._add_noise(imgs_gen_l, img_noise_scale)
        imgs_gen_m  = self.gan_m.get_gen_images(z, rand_perm=True)  
        imgs_gen_m = self._add_noise(imgs_gen_m, img_noise_scale)
        return imgs_gen_l, imgs_gen_m

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
    
    def get_clasf_cluster_probs(self, dataloader, classifiers, img_noise_scale=0): 
        '''Performs cluster inference over the entire training set w/ the 2 classifiers.
           Returns the avg. cluster probabilities between the 2 classifiers for each training example.'''
        
        dataloader=torch.utils.data.DataLoader(dataloader.dataset, batch_size=100, shuffle=False, drop_last=False)
        
        #empty sublists to accumulate the minibatches of probabilities 
        clasf_cluster_probs = [ [[] for _ in range(self.no_c_outputs)] for clasf in classifiers ]
        
        #iterates through the dataset to collect classifiers inference with minibatches
        for (batch_imgs, batch_targets) in dataloader:
            batch_imgs = batch_imgs.cuda()
            batch_imgs = self._add_noise(batch_imgs, img_noise_scale)        
            with torch.no_grad():
                clasf_cluster_probs_batch = [torch.exp(clasf(batch_imgs)).transpose(1,0) for clasf in classifiers]
            for i in range(len(classifiers)):
                for j in range(self.no_c_outputs):
                    clasf_cluster_probs[i][j].append(clasf_cluster_probs_batch[i][j])

        #concatenates results for each batch of the whole data 
        clasf_cluster_probs = [[torch.cat(batch).cpu().numpy() for batch in classifier_i_batches] 
                               for classifier_i_batches in clasf_cluster_probs]
        
        #gets the average between the two classifiers' cluster probabilities
        clasf_cluster_probs_avg = np.array([(clasf_cluster_probs[0][0] + clasf_cluster_probs[1][1])/2,
                                            (clasf_cluster_probs[0][1] + clasf_cluster_probs[1][0])/2])
        
        #gets parent node (k) probabilities by summing previous probabilities in l and m
        parent_cluster_probs = (self.dl_l.sampler.weights+self.dl_m.sampler.weights).numpy()
        
        #multiplies by the parent node`s probabilities 
        clasf_cluster_probs_avg[0] *= parent_cluster_probs
        clasf_cluster_probs_avg[1] *= parent_cluster_probs
        clasf_cluster_probs_avg = clasf_cluster_probs_avg.tolist()
        
        return clasf_cluster_probs_avg
    
    def _plot_img_grid(self, imgs_plot, img_save_path, node_name, text_logs_path):
        if imgs_plot.shape[1] == 3 or imgs_plot.cpu().shape[1] == 1:
            grid = make_grid(imgs_plot, nrow=20, normalize=True)
            if img_save_path is not None:
                save_image(grid, img_save_path)
                try:
                    print_save_log("\nSample of generated images from group {}:".format(node_name), text_logs_path)  
                    print_save_log('(This sample is saved at {})'.format(img_save_path), text_logs_path)   
                    display(Image(filename=img_save_path, width=900))
                except:
                    print_save_log('Jupyter image display not available for plotting sample of generated images from group {}'.format(node_name), text_logs_path)
                    #print_save_log("Sample of generated images from group {} saved at {}".format(node_name, img_save_path), text_logs_path)   
            else:
                print_save_log("\nNo image save path defined, can't save sample of generated images", text_logs_path)
        else:
            print_save_log("\nCan't plot/save imgs with shape {}".format(imgs_plot.shape), text_logs_path)        

    def view_log_headings(self, epoch, epochs, epoch_interval, text_logs_path, ref_it=-1, ref_attempt=-1):
        '''Part 1/4 of training logs'''

        log_headings = "[REFINEMENT %d OF NODE %s SPLIT] [EPOCH %d/%d] [EPOCH TIME INTERVAL: %.2f sec.] [REF %d] [ATTEMPT %d]"%(ref_it, self.node_k.name,
            epoch, epochs, epoch_interval, ref_it, ref_attempt)
        log_headings = get_log_heading(log_headings)
        print_save_log("\n\n" + log_headings, text_logs_path)


    def view_epoch_losses(self, epoch_metrics_dict_l, epoch_metrics_dict_m, text_logs_path):
        '''part 2/4 of training logs'''

        print_save_log("Mean epoch losses/acc for each component in group l's GAN", text_logs_path)
        print_save_log({k:np.round(v/self.no_batches,5) for k,v in epoch_metrics_dict_l.items()}, text_logs_path)  
        print_save_log("Mean epoch losses/acc for each component in group m's GAN:", text_logs_path)
        print_save_log({k:np.round(v/self.no_batches,5) for k,v in epoch_metrics_dict_m.items()}, text_logs_path)

    def view_gen_imgs(self, epoch, ref_attempt, refinement_path, text_logs_path):
        '''part 3/4 of training logs'''

        imgs_plot_l, imgs_plot_m = self.get_gen_images(img_noise_scale=0, batch_size=10)
        if self.node_k is not None:
            if refinement_path is not None:
                img_save_path_l = refinement_path + "attempt_{}_ep_{}_{}.jpg".format(ref_attempt, epoch, self.node_l.name)
                img_save_path_m = refinement_path + "attempt_{}_ep_{}_{}.jpg".format(ref_attempt, epoch, self.node_m.name)
        self._plot_img_grid(imgs_plot_l, img_save_path_l, "l", text_logs_path)
        self._plot_img_grid(imgs_plot_m, img_save_path_m, "m", text_logs_path)
    
    def verify_collapsed_generators(self, epoch, text_logs_path, img_noise_scale=0, collapse_check_loss=0.01, collapse_check_epoch=50, batch_size=100):
        '''part 4/4 of training logs'''

        if epoch < collapse_check_epoch:
            print_save_log("\nGenerator collapse will be checked after epoch {}".format(collapse_check_epoch), text_logs_path)
        else:
            print_save_log("\nChecking if generators have collapsed...", text_logs_path)
            imgs_gen_l, imgs_gen_m = self.get_gen_images(img_noise_scale, batch_size)
            losses_l = self.gan_l.get_disc_losses_for_gen(imgs_gen_l)
            losses_m = self.gan_m.get_disc_losses_for_gen(imgs_gen_m)
            
            for loss in losses_l + losses_m:
                if loss < collapse_check_loss and epoch>=collapse_check_epoch:
                    log_string = "\nDiscriminator loss for generated images is too low (<{}), indicating generators collapse. The training shall restart.".format(collapse_check_loss)
                    print_save_log(log_string, text_logs_path)
                    self.cancel_training_flag = True
                    break
            if not(self.cancel_training_flag):
                print_save_log("Generators collapse not found, the training shall continue.", text_logs_path)


    def view_new_clasf_clustering(self, new_cluster_prob_mass_per_class, ref_it, text_logs_path):        
        '''Prints logs with refined binary clustering result for current nodes'''
        
        #header
        log_headings = "REFINED BINARY CLUSTERING FOR NODE {} SPLIT OBTAINED WITH AVG CLASSIFIER'S INFERENCE".format(self.node_k.name)     
        log_headings = get_log_heading(log_headings)
        print_save_log("\n\n"+log_headings, text_logs_path)
        
        #clustering table
        print_save_log("Local binary soft clustering (prob. mass division) for node {} split after refinement, according to each class.".format(self.node_k.name), text_logs_path)
        print_save_log('Probability mass variation since last refinement or raw split is indicated in parenthesis for each cluster and class.', text_logs_path)
        print('(This table is saved at {})'.format(text_logs_path))   
        table_df = get_classification_table_variation(self.cluster_prob_mass_per_class, new_cluster_prob_mass_per_class, self.idx_to_class_dict, 
                                           node=self.node_k, table_name = 'Local split soft clusters refined')  
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

def refinement(args, dataloader_l, dataloader_m, epochs, noise_start, ref_it = -1, sample_interval=10, collapse_check_loss=0.001, 
                 save=False, node_k=None, print_vars=False):

    redo_training = True
    ref_attempt = 1
    max_attempts = 4

    while redo_training:

        #configure log saving paths
        refinement_path = os.path.join(node_k.node_path, "refinement_{}/".format(ref_it))
        os.makedirs(refinement_path, exist_ok=True)
        text_logs_path = refinement_path + "attempt_{}_training_logs.txt".format(ref_attempt)
        save_log_text('', text_logs_path, open_mode='w')

        #print main log headings
        log_headings = 'REFINEMENT {} OF OF NODE {} SPLIT (ATTEMPT {})'.format(ref_it, node_k.name, ref_attempt)
        log_headings = get_log_heading(log_headings, spacing=2)
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
        [gan_l, gan_m] = create_gans(args, no_gans=2)

        #print models' architecture
        log_headings = get_log_heading("MODELS ARCHITETURE")
        print_save_log('\n\n' + log_headings, text_logs_path)        
        print_save_log("Discriminator Architecture:", text_logs_path)
        print_save_log(gan_l.disc, text_logs_path)
        print_save_log("\nGernerator Architecture:", text_logs_path)
        print_save_log(gan_l.gen_set.paths[0], text_logs_path)      
        print_save_log("\nClassifier Architecture:", text_logs_path)
        print_save_log(gan_l.clasf, text_logs_path)

        trainer = GANGroupsTrainer(dataloader_l, 
                                   dataloader_m, 
                                   gan_l, 
                                   gan_m, 
                                   amp_enable=args.amp_enable,
                                   prior_distribution = "uniform",
                                   node_k = node_k)

        trainer.train(epochs = epochs, 
                      text_logs_path=text_logs_path,
                      refinement_path = refinement_path,
                      noise_start=noise_start, 
                      collapse_check_loss=collapse_check_loss, 
                      collapse_check_epoch=args.collapse_check_epoch, 
                      sample_interval=sample_interval, 
                      ref_it=ref_it, 
                      batch_size_gen=args.batch_size_gen,
                      ref_attempt = ref_attempt,
                      no_refinements=args.no_refinements)

        #flag for restarting the training if generation collapse is detected
        if trainer.cancel_training_flag == False:
            redo_training = False
        else:
            ref_attempt += 1
        if ref_attempt>max_attempts:
            max_attempt_log_headings = get_log_heading("LIMIT OF {} FAILED ATTEMPTS REACHED".format(max_attempts))
            max_attempt_log = "The training for this refinement reached the limit of {} failed attempts due generation collapse.".format(max_attempts)
            max_attempt_log += " Please, select more stable tunnings for the models so that the generation stops collapsing."
            print_save_log("\n\n" + max_attempt_log_headings, text_logs_path)
            print_save_log(max_attempt_log, text_logs_path)
            sys.exit(max_attempt_log)

    return trainer
        
        
