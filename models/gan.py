#torch imports
from torch.autograd import Variable
import torch
import numpy as np

class GAN:
    def __init__(self,
                 gen_set,
                 disc,
                 clasf,
                 feature_layers,
                 optimizer_G,
                 optimizer_D,
                 optimizer_C,
                 diversity_parameter_g
                ):
        '''Class for coordinating batch-wise update between the components of MGAN (raw split) or a GAN group (refinement) 

        gen_set (torch.nn.Module): generator(s)
        disc (torch.nn.Module): discriminator
        clasf (torch.nn.Module): classifier
        feature_layers (torch.nn.Module): shared feature extractor for classifier and discriminator
        optimizer_G (torch.optim.Adam): Adam optimizer for generator(s)
        optimizer_D (torch.optim.Adam): Adam optimizer for discriminator
        optimizer_C (torch.optim.Adam): Adam optimizer for classifier
        diversity_parameter_g (float): hyperparameter for weighting generators' classification loss component 
        '''

        #components
        self.gen_set = gen_set
        self.disc = disc
        self.clasf = clasf
        self.feature_layers = feature_layers
        self.latent_dim = gen_set.paths[0].latent_dim
        self.diversity_parameter_g = diversity_parameter_g

        #optimizers
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizer_C = optimizer_C
        
        #losses
        self.loss_disc = torch.nn.BCEWithLogitsLoss()
        self.loss_clasf = torch.nn.NLLLoss()        
        self.amp_enable = False
    
        self.metrics_dict = {'loss_disc_real': 0, 
                             'acc_disc_real' : 0,
                             'loss_disc_fake': 0, 
                             'acc_disc_fake': 0,
                             'loss_gen_disc': 0,
                             'loss_gen_clasf': 0,
                             'loss_clasf': 0,
                             'acc_clasf' : 0,
                             }
        self.Tensor = torch.cuda.FloatTensor 

    def bin_accuracy(self, pred, labels):
        corrects = (labels == torch.sigmoid(pred).round()).detach()
        acc = corrects.sum()/len(corrects)
        return acc

    def categorical_accuracy(self, pred, labels):
        corrects = (labels == torch.argmax(pred, dim = -1)).detach()
        acc = corrects.sum()/len(corrects)
        return acc

    def assign_amp(self, amp_autocast, amp_scaler):
        self.amp_autocast = amp_autocast
        self.amp_scaler = amp_scaler
    
    def enable_amp(self, amp_enable):
        self.amp_enable = amp_enable

    def train_on_batch(self, imgs_real, imgs_gen):
        '''Performs one iteration of update for Discriminator, Generators and Classifier (Raw Split training)

        imgs_real (torch.cuda.FloatTensor): mini-batch of real dataset images
        imgs_gen (torch.cuda.FloatTensor): mini-batch of generated images
        '''

        self.gen_set.train()
        self.disc.train()

        #classification labels
        labels_c = []
        labels_c.append(self.Tensor([0]*(imgs_gen.shape[0]//2)))   
        labels_c.append(self.Tensor([1]*(imgs_gen.shape[0]//2))) 
        labels_c = torch.cat(labels_c, dim=0).type(torch.cuda.LongTensor)
        
        #adversarial game labels
        labels_d_valid = Variable(self.Tensor(imgs_real.shape[0], 1).fill_(1.0), requires_grad=False)
        labels_d_fake  = Variable(self.Tensor(imgs_gen.shape[0], 1).fill_(0.0), requires_grad=False)
        labels_g_valid = Variable(self.Tensor(imgs_gen.shape[0], 1).fill_(1.0), requires_grad=False)

        # --------------------
        #  Train Discriminator
        # --------------------
        self.optimizer_D.zero_grad()
        with self.amp_autocast(self.amp_enable):

            #gets real images loss/acc
            validity = self.disc(imgs_real)
            loss_disc_real = self.loss_disc(validity, labels_d_valid)
            acc_disc_real = self.bin_accuracy(validity, labels_d_valid)

            #gets generated images loss/acc
            validity = self.disc(imgs_gen.detach())
            loss_disc_fake = self.loss_disc(validity, labels_d_fake)
            acc_disc_fake = self.bin_accuracy(validity, labels_d_fake)

            #gets total loss for discriminator
            loss_disc = loss_disc_fake + loss_disc_real 

        self.amp_scaler.scale(loss_disc).backward()
        self.amp_scaler.step(self.optimizer_D)

        # -----------------
        #  Train Classifier
        # -----------------
        self.optimizer_C.zero_grad()
        with self.amp_autocast(self.amp_enable):
            for par in self.feature_layers.parameters():
                par.requires_grad_(False)

            #gets classification loss/acc
            classification = self.clasf(imgs_gen.detach())
            loss_clasf = self.loss_clasf(classification, labels_c)
            acc_clasf = self.categorical_accuracy(classification, labels_c)

            for par in self.feature_layers.parameters():
                par.requires_grad_(True)
        self.amp_scaler.scale(loss_clasf).backward()
        self.amp_scaler.step(self.optimizer_C)
    
        # -----------------
        #  Train Generators
        # -----------------
        self.optimizer_G.zero_grad()
        with self.amp_autocast(self.amp_enable):
            
            #gets discriminative loss/acc
            imgs_ft_gen = self.feature_layers(imgs_gen)
            validity = self.disc(imgs_ft_gen, feature_input=True)
            loss_gen_disc = self.loss_disc(validity, labels_g_valid)

            #gets classification loss/acc
            classification = self.clasf(imgs_ft_gen, feature_input=True)
            if self.diversity_parameter_g > 0:
                loss_gen_clasf = self.loss_clasf(classification, labels_c)*self.diversity_parameter_g 

            #gets total loss for generators
            loss_gen = loss_gen_disc + loss_gen_clasf*self.diversity_parameter_g 

        self.amp_scaler.scale(loss_gen).backward()
        self.amp_scaler.step(self.optimizer_G)           
 
        #updates metrics dictionaries 
        self.metrics_dict['loss_disc_real'] = loss_disc_real.item()
        self.metrics_dict['acc_disc_real']  =  acc_disc_real.item()
        self.metrics_dict['loss_disc_fake'] = loss_disc_fake.item()
        self.metrics_dict['acc_disc_fake']  = acc_disc_fake.item()
        self.metrics_dict['loss_gen_disc'] = loss_gen_disc.item()
        self.metrics_dict['loss_gen_clasf'] = loss_gen_clasf.item()
        self.metrics_dict['loss_clasf'] = loss_clasf.item()
        self.metrics_dict['acc_clasf'] = acc_clasf.item()
        
        return self.metrics_dict

    def train_on_batch_refinement(self, imgs_real, imgs_gen_internal, imgs_gen_external=None, clasf_external=None):
        '''Performs one iteration of update for internal discriminator, internal generator, and internal classifier,
        also requiring external generator's data and external classifier (Refinement training)

        imgs_real (torch.cuda.FloatTensor): mini-batch of real dataset images
        imgs_gen_internal (torch.cuda.FloatTensor): mini-batch of generated images by the internal generator
        imgs_gen_external (torch.cuda.FloatTensor): mini-batch of generated images by the external generator for internal classifier's training
        clasf_external (torch.nn.Module): external classifier used by internal generator's training
        '''

        self.gen_set.train()
        self.disc.train()
        
        #classification labels
        labels_c = []
        labels_c.append(self.Tensor([0]*imgs_gen_internal.shape[0]))    
        labels_c.append(self.Tensor([1]*imgs_gen_external.shape[0]))
        labels_c = torch.cat(labels_c, dim=0).type(torch.cuda.LongTensor)
        
        #adversarial labels
        labels_d_valid = Variable(self.Tensor(imgs_real.shape[0], 1).fill_(1.0), requires_grad=False)
        labels_d_fake  = Variable(self.Tensor(imgs_gen_internal.shape[0], 1).fill_(0.0), requires_grad=False)
        labels_g_valid = Variable(self.Tensor(imgs_gen_internal.shape[0], 1).fill_(1.0), requires_grad=False)
    
        # --------------------
        #  Train Discriminator
        # --------------------
        loss_disc_fake = self.Tensor([0]) 
        loss_disc_real = self.Tensor([0]) 
        acc_disc_real = self.Tensor([0]) 
        acc_disc_fake = self.Tensor([0]) 
        self.optimizer_D.zero_grad()
        with self.amp_autocast(self.amp_enable):

            #real images result
            validity = self.disc(imgs_real)
            loss_disc_real = self.loss_disc(validity, labels_d_valid)
            acc_disc_real = self.bin_accuracy(validity, labels_d_valid)

            #gen images result
            validity = self.disc(imgs_gen_internal.detach())
            loss_disc_fake = self.loss_disc(validity, labels_d_fake)
            acc_disc_fake = self.bin_accuracy(validity, labels_d_fake)
            
            #total loss
            loss_disc = loss_disc_fake + loss_disc_real
        
        self.amp_scaler.scale(loss_disc).backward()
        self.amp_scaler.step(self.optimizer_D)

        # -----------------
        #  Train Classifier
        # -----------------
        self.optimizer_C.zero_grad()
        with self.amp_autocast(self.amp_enable):
            for par in self.feature_layers.parameters():
                par.requires_grad_(False)

            #gets classification  
            classification_internal = self.clasf(imgs_gen_internal.detach())
            classification_external = self.clasf(imgs_gen_external.detach())
            classification_concat = torch.cat([classification_internal, classification_external])

            #gets loss/acc
            loss_clasf = self.loss_clasf(classification_concat, labels_c) 
            acc_clasf = self.categorical_accuracy(classification_concat, labels_c)

            for par in self.feature_layers.parameters():
                par.requires_grad_(True)
                        
        self.amp_scaler.scale(loss_clasf).backward()
        self.amp_scaler.step(self.optimizer_C)

        # -----------------
        #  Train Generators
        # -----------------
        loss_gen_disc = self.Tensor([0])
        loss_gen_clasf = self.Tensor([0])
        
        self.optimizer_G.zero_grad()
        
        with self.amp_autocast(self.amp_enable):
            
            #gets discriminative loss/acc
            imgs_ft_gen_internal = self.feature_layers(imgs_gen_internal)
            validity = self.disc(imgs_ft_gen_internal, feature_input=True)
            loss_gen_disc = self.loss_disc(validity, labels_g_valid)

            #gets discriminative loss/acc
            classification_internal = self.clasf(imgs_ft_gen_internal, feature_input=True)
            if clasf_external.feature_layers == self.clasf.feature_layers:
                classification_external = clasf_external(imgs_ft_gen_internal, feature_input=True)
            else:
                classification_external = clasf_external(imgs_gen_internal)
            classification_concat = torch.cat([classification_internal, classification_external] )
            if self.diversity_parameter_g > 0:
                loss_gen_clasf = self.loss_clasf(classification_concat, labels_c)*self.diversity_parameter_g 
            
            loss_gen = loss_gen_disc + loss_gen_clasf
        
        self.amp_scaler.scale(loss_gen).backward()
        self.amp_scaler.step(self.optimizer_G)           
    
        self.metrics_dict['loss_disc_real'] = loss_disc_real.item()
        self.metrics_dict['acc_disc_real']  =  acc_disc_real.item()
        self.metrics_dict['loss_disc_fake'] = loss_disc_fake.item()
        self.metrics_dict['acc_disc_fake']  =  acc_disc_fake.item()
        self.metrics_dict['loss_gen_disc'] = loss_gen_disc.item()
        self.metrics_dict['loss_gen_clasf'] = loss_gen_clasf.item()
        self.metrics_dict['loss_clasf'] = loss_clasf.item()
        self.metrics_dict['acc_clasf'] = acc_clasf.item()
        
        return self.metrics_dict
    
    def get_gen_images(self, z, rand_perm=False):
        return(self.gen_set(z, rand_perm=rand_perm))
    
    def get_disc_losses_for_gen(self, imgs_gen_internal, no_generators=2):
        self.disc.train()
        batch_size = imgs_gen_internal.shape[0]//no_generators
        fake  = Variable(self.Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
        losses_for_gen = []
        for i in range(no_generators):
            imgs_gen_i = imgs_gen_internal[batch_size*i:batch_size*(i+1)]
            with torch.no_grad():
                validity = self.disc(imgs_gen_i.detach())
                loss_fake = self.loss_disc(validity, fake).detach()
                losses_for_gen.append(loss_fake.item())
        return losses_for_gen
