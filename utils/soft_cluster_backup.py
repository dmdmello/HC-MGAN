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

def print_save_log(log, save_path=None, print_log=True):
    if print_log:
        print(log)
    if save_path is not None: 
        save_log_text(remove_bold_from_string(str(log))+'\n', save_path, open_mode='a')


def view_global_tree_logs(dataloader_train, non_leaf_nodes, leaf_nodes, no_classes=10, log_save_path='', log_title = None, filter_classes=[], display_table=True):

    if log_title is None:
        log_title = '\nGLOBAL TREE LOGS AFTER LAST RAW SPLIT OR REFINEMENT'
    log_heading = get_bold_string(get_log_heading(log_title, spacing=2))
    print_save_log('\n\n\n'+log_heading, log_save_path)
    
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
    cluster_counts_per_class = get_hard_cluster_per_class_parallel(dataloader_train, leaf_nodes_probs, no_of_classes=no_classes, filter_classes = filter_classes)
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

    #ACC
    print_save_log("\n(Observation on the following ACC metrics: if the numbers of classes and clusters don't match, there are multiple ways to compute ACC." +
    " Each value bellow considers constraints on the max N classes allowed assigned to each available cluster," + 
    " with correct examples given by the optimum assignment according to each constraint, and the ACC given by the total of correct examples divided by the total of examples." +
    " When the numbers of classes and clusters are equal, max 1 ACC provides the usual ACC present in horizontal clustering benchmarks)", log_save_path)
    max_classes_per_cluster = 1
    '''while(max_classes_per_cluster < no_classes):
        print_opt_assignment_acc_logs(max_classes_per_cluster, cluster_counts_per_class, log_save_path, classes_names_dict, leaf_nodes)
        max_classes_per_cluster += 1'''
    classes_per_cluster_best = []
    classes_counts_per_cluster_best = []
    total_counts_best = 0
    for max_classes_per_cluster in range(no_classes):
        classes_per_cluster, classes_counts_per_cluster = get_opt_assignment(max_classes_per_cluster, cluster_counts_per_class)
        if np.sum(classes_counts_per_cluster)>total_counts_best:
            classes_per_cluster_best = classes_per_cluster
            classes_counts_per_cluster_best = classes_counts_per_cluster_best

def get_opt_assignment(max_classes_per_cluster, cluster_counts_per_class):
    """Gets optimum cluster assignment with hungarian algorithm, returning classes assignments and classes counts per cluster.
    For enabling multiple classes per cluster, the clustering matrix needs to have its cluster idx (columns) replicated n times,
    where n is the maximum allowed classes allowed for each cluster.
    
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

def print_opt_assignment_acc_logs(max_classes_per_cluster, cluster_counts_per_class, log_save_path, classes_names_dict, leaf_nodes):
    mat = np.repeat(cluster_counts_per_class, max_classes_per_cluster, axis=1)
    opt_assign_lines, opt_assign_columns = scipy.optimize.linear_sum_assignment(mat, maximize=True)
    opt_assign_counts_per_cluster = np.array(mat)[opt_assign_lines, opt_assign_columns]
    opt_assign_total_corrects =  opt_assign_counts_per_cluster.sum()
    total_examples = np.array(cluster_counts_per_class).sum()
    acc = opt_assign_total_corrects/total_examples
    print_save_log('\nACC with max {} classes per cluster: {}/{} = {}'.format(max_classes_per_cluster, opt_assign_total_corrects, total_examples, acc), log_save_path)
    opt_assign_string = 'Optimum assignment: '
    no_clusters = len(cluster_counts_per_class[0])
    for i in range(no_clusters):
        classes_cluster_i = opt_assign_lines[opt_assign_columns//max_classes_per_cluster==i]
        correct_examples_cluster_i = opt_assign_counts_per_cluster[opt_assign_columns//max_classes_per_cluster==i]
        opt_assign_string += '['
        opt_assign_string += ",".join(["'"+classes_names_dict[c]+"'({})".format(c_counts) 
                                for c,c_counts in zip(classes_cluster_i, correct_examples_cluster_i)])
        opt_assign_string += ']'
        opt_assign_string += " --> '{}'; ".format(leaf_nodes[i].name)
    print_save_log(opt_assign_string, log_save_path)


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
    
def print_table(table_dict):

    df = pd.DataFrame(table_dict)
    dfStyler = df.style.set_properties(**{'text-align': 'center', 
                                          'border' : '1px  solid !important' })
    df = dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    display(df)



def get_most_associeted_classes_per_clusters(clusters_per_class, max_classes_per_group = 2):

    no_clusters = len(clusters_per_class[0])
    assert(no_clusters == 2)
    no_classes = len(clusters_per_class)
    
    most_associeted_classes_per_cluster_labels = [[], []]
    most_associeted_classes_per_cluster_counts = [[], []]
    remaining_groups = []
    
    for i in range(no_classes):
        
        if clusters_per_class[i][0] > clusters_per_class[i][1]:
            most_associeted_classes_per_cluster_labels[0].append(i)
            most_associeted_classes_per_cluster_counts[0].append(clusters_per_class[i][0])
        else:
            most_associeted_classes_per_cluster_labels[1].append(i)
            most_associeted_classes_per_cluster_counts[1].append(clusters_per_class[i][1])

    while len(most_associeted_classes_per_cluster_labels[0])>max_classes_per_group:
        
        least_i = np.argmin(most_associeted_classes_per_cluster_counts[0])
        most_associeted_classes_per_cluster_labels[1] += [most_associeted_classes_per_cluster_labels[0][least_i]]
        most_associeted_classes_per_cluster_counts[0].pop(least_i)
        most_associeted_classes_per_cluster_labels[0].pop(least_i)
        
    while len(most_associeted_classes_per_cluster_labels[1])>max_classes_per_group:
        
        least_i = np.argmin(most_associeted_classes_per_cluster_counts[1])
        most_associeted_classes_per_cluster_labels[0] += [most_associeted_classes_per_cluster_labels[1][least_i]]
        most_associeted_classes_per_cluster_counts[1].pop(least_i)
        most_associeted_classes_per_cluster_labels[1].pop(least_i)
    
    return most_associeted_classes_per_cluster_labels

def merge_classes(clusters_per_class, merged_classes_targets_groups=None):
    
    if merged_classes_targets_groups is None:
        merged_classes_targets_groups = [[i] for i in range(10)]

    no_of_clusters = len(clusters_per_class[0])
    no_of_classes = len(merged_classes_targets_groups)
    
    merged_classes_counts = []

    for class_targets_group in merged_classes_targets_groups:
        group_counts = np.zeros(no_of_clusters, dtype=type(clusters_per_class[0][0]))
        for class_target in class_targets_group:
            group_counts += np.array(clusters_per_class[class_target])

        merged_classes_counts.append(group_counts.tolist())
        
    return merged_classes_counts

def get_maxN_classes_acc(clusters_per_class, maxN = 2):
    
    most_associeted_classes_per_cluster = get_most_associeted_classes_per_clusters(clusters_per_class, maxN)

    clusters_per_class = merge_classes(clusters_per_class, most_associeted_classes_per_cluster)
    
    assert(len(clusters_per_class)==2)
    
    total_classes = np.sum(clusters_per_class)
    
    correct_classes_left = clusters_per_class[0][0]
    correct_classes_right = clusters_per_class[1][1]
    
    acc = (correct_classes_left + correct_classes_right)/total_classes
    
    return acc


def get_maxN_cluster_purity(clusters_per_class, maxN = 2):
    
    classes_per_cluster = np.transpose(clusters_per_class).tolist()
    cluster_left = classes_per_cluster[0]
    cluster_right = classes_per_cluster[1]
    total_elements = np.sum(classes_per_cluster)

    total_top_classes_in_left = 0
    for i in range(maxN):
        top_i_class_idx = np.argmax(cluster_left)
        total_top_classes_in_left += cluster_left[top_i_class_idx]
        cluster_left.pop(top_i_class_idx)

    total_top_classes_in_right = 0
    for i in range(maxN):
        top_i_class_idx = np.argmax(cluster_right)
        total_top_classes_in_right += cluster_right[top_i_class_idx]
        cluster_right.pop(top_i_class_idx)

    weighted_purity = (total_top_classes_in_left+total_top_classes_in_right)/total_elements
    
    return weighted_purity

import sklearn 

def get_total_corrects(clusters_per_class):
    total_corrects = 0
    for clusters in clusters_per_class:
        total_corrects += max(clusters)
    return total_corrects


def get_double_clustering_metrics(clusters_per_class, maxN=2, classes_for_monitoring=None):
    
    assert(len(clusters_per_class[0])==2)
    
    if classes_for_monitoring:
        clusters_per_class = [clusters_per_class[i] for i in classes_for_monitoring]
    
    reference_labels = []
    
    for i in range(len(clusters_per_class)):
        reference_labels += [i]*np.array(clusters_per_class[i]).sum() 
    
    clustering_labels = []
    for i in  range(len(clusters_per_class)):
        clustering_labels += [0]*clusters_per_class[i][0]
        clustering_labels += [1]*clusters_per_class[i][1]
    
    digits = 3
    no_classes = len(clusters_per_class)
    half = int(np.ceil(no_classes/2))
    maxN_acc_list = [np.round(get_maxN_classes_acc(clusters_per_class, i), digits) for i in range(half, no_classes)]
    max_N_acc_arg = np.argmax(maxN_acc_list)
    max_N_acc = maxN_acc_list[max_N_acc_arg]
    
    total_corrects = get_total_corrects(clusters_per_class)
    
    nmi = np.round(sklearn.metrics.cluster.normalized_mutual_info_score(reference_labels, clustering_labels), digits)
     
    metrics_dict = {'Max Possible Acc (max {} classes per cluster)'.format(max_N_acc_arg+half):max_N_acc, 
                    'NMI': nmi, 
                   }
    
    return metrics_dict

def get_local_cluster_table(clusters_per_class, classes_names_dict, table_name = 'Local binary clustering', node_name='UnknownNode'):
    
    no_of_classes = len(clusters_per_class)

    classes_names = [classes_names_dict[c] for c in range(no_of_classes)]
        
    table_dict = {}
    
    table_dict[table_name] = ["Left cluster: " + node_name + "L (Raw Split)", "Right cluster: " + node_name + "R (Raw Split)"]

    for i in range(no_of_classes):
                        
        column_name = classes_names[i] + '({})'.format(np.sum(clusters_per_class[i]).round(2))
        classes_names_dict[i]
        column_contents = clusters_per_class[i]
        table_dict[column_name] = column_contents
    
    return pd.DataFrame(table_dict)

def get_classification_table_variation(clusters_per_class_orig, clusters_per_class_new, classes_names_dict, node_k, data_prefix = '', 
                                       print_table=True, table_name='Clustering result', ref_it = 1):
    
    no_of_clusters = len(clusters_per_class_orig[0])
    no_of_classes = len(clusters_per_class_orig)
    
    classes_names = [classes_names_dict[t] for t in range(no_of_classes)]
        
    table_dict = {}
    
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    table_dict[table_name] = ["Left cluster: " + node_k.name + "L (" +ordinal(ref_it)+" Refinement)", 
                              "Right cluster: " + node_k.name + "R (" +ordinal(ref_it)+" Refinement)"]

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
    




'''def view_global_tree_logs(dataloader_train, non_leaf_nodes, leaf_nodes, no_classes=10, log_save_path='', log_title = None, filter_classes=[], display_table=True):
    #leaf_nodes = root_node.get_leaf_nodes_list()

    #log_headings = '\033[1mGLOBAL TREE LOGS AFTER LAST RAW SPLIT OR REFINEMENT\033[0m'
    #hyphen_bar = len(log_headings)*'-'
    #print ("\n"+hyphen_bar + '\n\n' +log_headings + '\n\n' + hyphen_bar)
    if log_title is None:
        log_title = 'GLOBAL TREE LOGS AFTER LAST RAW SPLIT OR REFINEMENT'
    print_save_log(get_bold_string(get_log_heading(log_title, spacing=2)), log_save_path)
    
    #LOG 1
    #print("\n"+hyphen_bar + '\n\033[1mLOG 1/3: TREE NODES\033[0m' + "\n" + hyphen_bar + "\n")
    print_save_log(get_bold_string(get_log_heading('GLOBAL TREE LOG 1/3: TREE NODES')), log_save_path)
    
    print_save_log('\nNon-leaf nodes:', log_save_path)
    print_save_log(str([node.name + ' (' + node.status + ')' for node in non_leaf_nodes]), log_save_path)

    print_save_log('\nLeaf nodes:', log_save_path)
    print_save_log(str([node.name + ' (' + node.status + ')' for node in leaf_nodes]), log_save_path)

    #cluster_probs = node.cluster_probs + node.cluster_probs_refinement
    #print_save_log("\n"+hyphen_bar + '\n\033[1mLOG 2/3: CLUSTERING MATRIX:\033[0m' + "\n" + hyphen_bar + "\n")

    #LOG 2
    print_save_log(get_bold_string(get_log_heading('GLOBAL TREE LOG 2/3: CLUSTERING MATRIX')), log_save_path)

    print_save_log("This table indicates the clustering matrix when it reaches N clusters, or N leaf nodes.", log_save_path)
    print_save_log("The final matrix occurs when N equals the number of classes.\n", log_save_path)
    leaf_nodes_probs = []
    for node in leaf_nodes:
        cluster_probs = node.cluster_probs
        leaf_nodes_probs.append(cluster_probs[-1].numpy())
    leaf_nodes_probs = np.array(leaf_nodes_probs)
    cluster_counts_per_class = get_hard_cluster_per_class_parallel(dataloader_train, leaf_nodes_probs, no_of_classes=no_classes, filter_classes = filter_classes)
    #print_save_log("\nClustering matrix (column=clusters, rows=classes) formed by leaf nodes:\n", np.array(cluster_counts_per_class))
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
        try:
            display(pd.DataFrame(cluster_matrix_dict))
            print_save_log(str(pd.DataFrame(cluster_matrix_dict)), log_save_path, print_log=False)
        except:
            print_save_log(str(pd.DataFrame(cluster_matrix_dict)), log_save_path)
    
    #LOG 3
    #print_save_log("\n"+hyphen_bar + '\n\033[1mLOG 3/3: CLUSTERING METRICS:\033[0m' + "\n" +  hyphen_bar + "\n")
    print_save_log(get_bold_string(get_log_heading('GLOBAL TREE LOG 3/3: CLUSTERING METRICS')), log_save_path)

    print_save_log("\nNormalized Mutual Information (NMI): {}".format(get_parallel_clustering_nmi(cluster_counts_per_class)), log_save_path)

    print_save_log("\n(Observation on the following ACC metric: if the numbers of classes and clusters don't match, there are multiple ways to compute ACC." +
    " Each value bellow considers constraints on the max N classes allowed assigned to each available cluster," + 
    " with correct examples given by the optimum assignment according to each constraint, and the ACC given by the total of correct examples divided by the total of examples." +
    " When the numbers of classes and clusters are equal, max 1 ACC provides the usual ACC present in horizontal clustering benchmarks)", log_save_path)

    n_repeat = 1
    while(len(leaf_nodes_probs)*n_repeat <= no_classes):
        mat = np.repeat(cluster_counts_per_class, n_repeat, axis=1)
        lin, col = scipy.optimize.linear_sum_assignment(mat, maximize=True)
        opt_right_clusters = np.array(mat)[lin, col]
        acc = opt_right_clusters.sum()/(np.sum(cluster_counts_per_class))

        print_save_log('\nACC with max {} classes per cluster: {}/{} = {}'.format(n_repeat, opt_right_clusters.sum(), np.sum(cluster_counts_per_class), acc), log_save_path)
        #print_save_log('optimum choice:'.format(n_repeat), opt_right_clusters)
        opt_assignment_string = 'Optimum choice: '
        for i in range(len(col)//n_repeat):
            classes_cluster_i = lin[col//n_repeat==i]
            correct_examples_cluster_i = opt_right_clusters[col//n_repeat==i]
            opt_assignment_string += '['
            opt_assignment_string += ",".join(["'"+classes_names_dict[c]+"'({})".format(c_counts) 
                                    for c,c_counts in zip(classes_cluster_i, correct_examples_cluster_i)])
            opt_assignment_string += ']'
            #opt_assignment_string += ",".join(["'"+classes_names_dict[x]+"'" for x in classes_cluster_i])
            #opt_assignment_string += "("
            #opt_assignment_string += "+".join([str(x) for x in correct_examples_cluster_i])
            #opt_assignment_string += " ex.) "
            opt_assignment_string += " --> '{}'; ".format(leaf_nodes[i].name)
        print_save_log(opt_assignment_string, log_save_path)
        #acc = opt_right_clusters.sum()/(np.sum(cluster_counts_per_class))
        n_repeat += 1'''

