import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import seaborn as sn
import pandas as pd

import numpy as np
import math
from sklearn import metrics
import sklearn
import scipy
import scipy.optimize as opt

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm_notebook

import torch.nn as nn
import torch.nn.functional as F
import torch
import shutil
import time

from utils.others import get_log_heading, get_bold_string, save_log_text, remove_bold_from_string, print_save_log

CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

def view_global_tree_logs(dataloader_train, non_leaf_nodes, leaf_nodes, log_save_path, log_title = None, display_table=True):

    if log_title is None:
        log_title = '\nGLOBAL TREE LOGS AFTER LAST RAW SPLIT OR REFINEMENT'
    log_heading = get_bold_string(get_log_heading(log_title, spacing=2))
    print_save_log('\n\n\n'+log_heading, log_save_path)
    no_classes = len(dataloader_train.dataset.classes)

    #-----------------------
    #LOGS PART 1: TREE NODES
    #-----------------------
    log_title = 'GLOBAL TREE LOGS 1/3: TREE NODES'
    log_heading = get_bold_string(get_log_heading(log_title))
    print_save_log(log_heading, log_save_path)
    print_save_log('Non-leaf nodes:', log_save_path)
    print_save_log(str([node.name + ' (' + node.status + ')' for node in non_leaf_nodes]), log_save_path)
    print_save_log('\nLeaf nodes:', log_save_path)
    print_save_log(str([node.name + ' (' + node.status + ')' for node in leaf_nodes]), log_save_path)
    
    #-------------------------------
    #LOGS PART 2: CLUSTERING MATRIX
    #-------------------------------
    log_title = 'GLOBAL TREE LOGS 2/3: CLUSTERING MATRIX'
    log_heading = get_bold_string(get_log_heading(log_title))
    print_save_log('\n\n'+log_heading, log_save_path)
    print_save_log("This table indicates the clustering matrix when it reaches N clusters, or N leaf nodes.", log_save_path)
    print_save_log("The final matrix occurs when N equals the number of classes.\n", log_save_path)
    print("(This table is saved at {})".format(log_save_path))
    leaf_nodes_probs = []
    for node in leaf_nodes:
        cluster_probs = node.cluster_probs
        leaf_nodes_probs.append(cluster_probs[-1].numpy())
    leaf_nodes_probs = np.array(leaf_nodes_probs)
    cluster_counts_per_class = get_hard_cluster_per_class_parallel(dataloader_train, leaf_nodes_probs, no_of_classes=no_classes)
    cluster_matrix = np.array(cluster_counts_per_class)
    cluster_matrix_dict = {}
    classes_names_dict = {v:k for k,v in dataloader_train.dataset.class_to_idx.items()}
    classes_names = [classes_names_dict[t] for t in range(no_classes)]
    cluster_matrix_dict['Leaf Nodes Clusters'] = [node.name + ' (' + node.status + ')' for node in leaf_nodes]
    for i in range(len(cluster_matrix)):
        column_name = classes_names[i] + '({})'.format(np.sum(cluster_matrix[i]).round(2))
        column_contents = cluster_matrix[i]
        cluster_matrix_dict[column_name] = column_contents
    if display_table:
        pd.set_option("max_colwidth", None)
        pd.set_option('max_columns', None)
        try:
            display(pd.DataFrame(cluster_matrix_dict))
            print_save_log(str(pd.DataFrame(cluster_matrix_dict)), log_save_path, print_log=False)
        except:
            print_save_log(str(pd.DataFrame(cluster_matrix_dict)), log_save_path)
    
    #-------------------------------
    #LOGS PART 3: CLUSTERING METRICS
    #-------------------------------
    log_title = 'GLOBAL TREE LOGS 3/3: CLUSTERING METRICS'
    log_heading = get_bold_string(get_log_heading(log_title))
    print_save_log('\n\n'+log_heading, log_save_path)
    
    #NMI
    nmi = get_parallel_clustering_nmi(cluster_counts_per_class)
    print_save_log("Normalized Mutual Information (NMI): {}".format(nmi), log_save_path)

    #max 1 class ACC
    classes_per_cluster, classes_counts_per_cluster = get_opt_assignment(1, cluster_counts_per_class)
    total_counts = np.sum([np.sum(classes_counts) for classes_counts in classes_counts_per_cluster])
    total_data_examples= np.sum(cluster_counts_per_class)
    acc = total_counts/total_data_examples
    print_save_log('\nBest accuracy (ACC) with at most 1 (one) class per cluster: {}/{} = {}'.format(total_counts, total_data_examples, acc), log_save_path)
    opt_assign_string = 'Optimum assignment considered: \n'
    opt_assign_string += get_opt_assignment_str(classes_per_cluster, classes_counts_per_cluster, classes_names_dict, leaf_nodes)
    print_save_log(opt_assign_string, log_save_path)

    #ACC
    classes_per_cluster_best = []
    classes_counts_per_cluster_best = []
    total_counts_best = 0
    for max_classes_per_cluster in range(1, no_classes+1):
        classes_per_cluster, classes_counts_per_cluster = get_opt_assignment(max_classes_per_cluster, cluster_counts_per_class)
        total_counts = np.sum([np.sum(classes_counts) for classes_counts in classes_counts_per_cluster])
        if total_counts>total_counts_best:
            classes_per_cluster_best = classes_per_cluster
            classes_counts_per_cluster_best = classes_counts_per_cluster
            total_counts_best = total_counts
    acc = total_counts_best/total_data_examples
    print_save_log('\nBest accuracy (ACC) with multiple classes per cluster: {}/{} = {}'.format(total_counts_best, total_data_examples, acc), log_save_path)
    opt_assign_string = 'Optimum assignment considered: \n'
    opt_assign_string += get_opt_assignment_str(classes_per_cluster_best, classes_counts_per_cluster_best, classes_names_dict, leaf_nodes)
    print_save_log(opt_assign_string, log_save_path)

    print_save_log("\n(Note on the above ACC metrics: if the no. of classes is less then the no. clusters, " + 
                   "we can either consider multiple classes belonging to a single cluster or left certain classes unassigned for computing ACC. " +
                   "The first ACC metric above considers at most 1 classes per cluster, and when the number of clusters and classes are equal, it provides the "+
                   "usual ACC metric used in horizontal clustering and also used in our paper as benchmark." + 
                   "The second ACC metric considers the best assignment possible with multiple classes allowed to be assigned to each cluster, " + 
                   "and its useful to track an upper bound for the final 1-to-1 ACC during the growth of the tree, before it reaches one cluster to each class.", log_save_path)

def get_opt_assignment(max_classes_per_cluster, cluster_counts_per_class):
    """Gets optimum cluster assignment with hungarian algorithm, returning classes assignments and classes counts per cluster.
    For enabling multiple classes per cluster, the clustering matrix needs to have its cluster idx (columns) replicated n times,
    where n is the maximum number of classes allowed for each cluster.
    
    Args:
    max_classes_per cluster (int): maximum classes allowed for each cluster during the search for optimum assignment
    cluster_counts_per_class (int list): clustering matrix with axis 0 relating to classes and axis 1 to clusters
    """
 
    #cluster matrix is repeated N times to allow max N classes per cluster
    mat = np.repeat(cluster_counts_per_class, max_classes_per_cluster, axis=1)

    #gets optimum assignment idxs and example counts
    lines, columns = scipy.optimize.linear_sum_assignment(mat, maximize=True)
    opt_assign_counts_per_cluster = np.array(mat)[lines, columns]

    #columns idxs refer to the N times repeated columns. 
    #to get cluster idxs, we need the integer division of the repeated idxs by their repetition number
    columns_as_cluster_idx = columns//max_classes_per_cluster

    #for loop for getting class idxs and class counts for each cluster i
    classes_per_cluster = []
    classes_counts_per_cluster = []
    no_clusters = len(cluster_counts_per_class[0])
    for i in range(no_clusters):
        classes_per_cluster.append(lines[columns_as_cluster_idx==i])
        classes_counts_per_cluster.append(opt_assign_counts_per_cluster[columns_as_cluster_idx==i])
    return classes_per_cluster, classes_counts_per_cluster

def get_opt_assignment_str(classes_per_cluster, classes_counts_per_cluster, classes_names_dict, leaf_nodes):
    no_clusters = len(classes_per_cluster)
    opt_assign_string = ''
    for i in range(no_clusters):
        opt_assign_string += '['
        opt_assign_string += ",".join(["'"+classes_names_dict[c]+"'({})".format(c_counts) 
                                for c,c_counts in zip(classes_per_cluster[i], classes_counts_per_cluster[i])])
        opt_assign_string += ']'
        opt_assign_string += " --> '{}'; ".format(leaf_nodes[i].name)
    return opt_assign_string


def get_hard_cluster_per_class_parallel(dataloader, split_cluster_probs, no_of_classes = 10, filter_classes=[]): 
    
    max_mask = (split_cluster_probs.max(axis=0,keepdims=1) == split_cluster_probs)
    #print(max_mask[0])
    no_of_clusters = len(split_cluster_probs)

    cluster_counts_per_class = []
    
    cluster_probs_sum = split_cluster_probs[0] + split_cluster_probs[1]
        
    for i in range(no_of_classes):
        cluster_counts_ij = []
        if i not in filter_classes: 
            for j in range(no_of_clusters):
                #print(j)
                cluster_counts_ij.append( (((np.array(dataloader.dataset.targets)==i))*np.array(max_mask[j]) ).sum().item() )

            cluster_counts_per_class.append(cluster_counts_ij)

    classes_names_dict = {v:k for k,v in dataloader.dataset.class_to_idx.items()}
    #print(np.array(cluster_counts_per_class))
    
    return cluster_counts_per_class

def get_parallel_clustering_nmi(cluster_counts_per_class):
    reference_labels = []
    for i in range(len(cluster_counts_per_class)):
        reference_labels += [i]*np.array(cluster_counts_per_class[i]).sum() 

    clustering_labels = []
    for i in  range(len(cluster_counts_per_class)):
        for  j in range(len(cluster_counts_per_class[0])):    
            clustering_labels += [j]*cluster_counts_per_class[i][j]
    #print(len(reference_labels))
    #print(len(clustering_labels))
    nmi = sklearn.metrics.cluster.normalized_mutual_info_score(reference_labels, clustering_labels)
    return nmi


def show(img, rows):
    npimg = img.detach().numpy()
    plt.figure(figsize = (20, rows))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    plt.show()

def distribution_select(dist, shape):
    
    assert(dist in ['uniform', 'normal'])
    
    if dist=='uniform':
        return np.random.uniform(-1, 1, shape)
    elif dist=='normal':
        return np.random.normal(0, 1, shape)
    else: 
        return None


def get_local_cluster_table(clusters_per_class, classes_names_dict, node, table_name = 'Local binary clustering'):
    
    no_of_classes = len(clusters_per_class)

    classes_names = [classes_names_dict[c] for c in range(no_of_classes)]
        
    table_dict = {}
    
    left = node.child_left
    right = node.child_right

    table_dict[table_name] = ["Left cluster: " + left.name + "({})".format(left.status), "Right cluster: " + right.name + "({})".format(right.status)]

    for i in range(no_of_classes):
                        
        column_name = classes_names[i] + '({})'.format(np.sum(clusters_per_class[i]).round(2))
        classes_names_dict[i]
        column_contents = clusters_per_class[i]
        table_dict[column_name] = column_contents
    
    return pd.DataFrame(table_dict)

def get_classification_table_variation(clusters_per_class_orig, clusters_per_class_new, classes_names_dict, node, data_prefix = '', table_name='Clustering result'):
    
    no_of_clusters = len(clusters_per_class_orig[0])
    no_of_classes = len(clusters_per_class_orig)
    
    classes_names = [classes_names_dict[t] for t in range(no_of_classes)]
        
    table_dict = {}

    left = node.child_left
    right = node.child_right

    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    table_dict[table_name] = ["Left cluster: " + left.name + "({})".format(left.status), 
                              "Right cluster: " + right.name + "({})".format(right.status)]

    clusters_per_class_diff = np.array(clusters_per_class_new) - np.array(clusters_per_class_orig)
    clusters_per_class_diff = clusters_per_class_diff.round(2)
    
    for i in range(no_of_classes):
        
        column_name = data_prefix + classes_names[i] + '({})'.format(np.sum(clusters_per_class_new[i]).round(2))
        column_contents_new = clusters_per_class_new[i]
        column_contents_diff = clusters_per_class_diff[i]
        column_formatted = ['{} (+{})'.format(column_contents_new[j], column_contents_diff[j]) if column_contents_diff[j]>=0 
                            else '{} ({})'.format(column_contents_new[j], column_contents_diff[j])  for j in range(len(clusters_per_class_new[i])) ]
        
        table_dict[column_name] = column_formatted
    
    return(pd.DataFrame(table_dict))
    