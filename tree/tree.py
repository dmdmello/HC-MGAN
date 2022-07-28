import torch 
from tree.refinement import refinement
from tree.raw_split import raw_split
import numpy as np
import copy
import os
from utils.soft_cluster import view_global_tree_logs, show, view_global_tree_logs
from utils.others import save_log_text, remove_bold_from_string, print_save_log, get_log_heading

class Node:
    def __init__(self,
                 name,
                 cluster_probs,
                 tree_path,
                 parent=None,
                 child_left=None,
                 child_right=None,
                 node_status = "root node"):
        
        self.name = name
        self.cluster_probs = [cluster_probs]
        self.parent = parent
        self.child_left = child_left
        self.child_right = child_right
        self.tree_path = tree_path
        self.node_path = os.path.join(tree_path, name)   
        self.status = node_status
        self.skipped_refinemnts = False    

    def add_cluster_probs(self, cluster_probs):
        self.cluster_probs.append(cluster_probs)

    def create_children(self, cluster_probs_left, cluster_probs_right):
        self.child_left = Node(self.name + 'L', torch.Tensor(cluster_probs_left), tree_path=self.tree_path, parent=self, node_status='raw split')
        self.child_right = Node(self.name + 'R', torch.Tensor(cluster_probs_right), tree_path=self.tree_path, parent=self, node_status='raw split')

def get_leaf_nodes_list(root_node):
    leaf_nodes_list = []

    if (root_node.child_left is None) and (root_node.child_right is None): 
        return [root_node]
    else:
        if root_node.child_left is not None:
            leaf_nodes_list += (get_leaf_nodes_list(root_node.child_left))

        if root_node.child_right is not None:
            leaf_nodes_list += (get_leaf_nodes_list(root_node.child_right))

        return leaf_nodes_list

def get_non_leaf_nodes_list(root_node): 

    if (root_node.child_left is None) and (root_node.child_right is None): 
        return []
    else:
        non_leaf_nodes_list = [root_node]

        if root_node.child_left is not None:
            non_leaf_nodes_list += (get_non_leaf_nodes_list(root_node.child_left))

        if root_node.child_right is not None:
            non_leaf_nodes_list += (get_non_leaf_nodes_list(root_node.child_right))

        return non_leaf_nodes_list


def get_node_by_name(root_node, name):
    leaf_nodes_list = get_leaf_nodes_list(root_node)
    non_leaf_nodes_list = get_non_leaf_nodes_list(root_node)

    for node in leaf_nodes_list:
        if node.name == name:
            print("Node '{}' was found".format(name))
            return node

    for node in non_leaf_nodes_list:
        if node.name == name:
            print("Node '{}' was found".format(name))
            return node

    print("Node '{}' was not found".format(name))
    return None

def search_node_to_split(root_node, text_logs_path):
    log_headings = get_log_heading("SEARCHING NEXT LEAF NODE TO SPLIT", spacing=2)
    print_save_log('\n\n\n' + log_headings, text_logs_path)        
    leaf_nodes_list = get_leaf_nodes_list(root_node)
    prob_mass_per_leaf = [leaf.cluster_probs[-1].sum() for leaf in leaf_nodes_list]
    split_node = leaf_nodes_list[np.argmax(prob_mass_per_leaf)]
    print_save_log('Currently {} leaf nodes obtained: '.format(len(leaf_nodes_list)), text_logs_path) 
    print_save_log([(node.name, '{} prob. mass'.format(node.cluster_probs[-1].sum())) for node in leaf_nodes_list], text_logs_path)
    log = 'Selecting for split leaf node {} (prob. mass {}) following the greatest prob. mass criteria.'.format(split_node.name, split_node.cluster_probs[-1].sum())
    print_save_log(log, text_logs_path)
    return split_node

def raw_split_tree_node(args, node_k, dataloader_train, halt_epoch= 20,  collapse_check_loss=0.01, save_node_path=None):
    
    dataloader_cluster_k = copy.deepcopy(dataloader_train)
    dataloader_cluster_k.sampler.weights = node_k.cluster_probs[-1]
    
    if node_k.node_path is not None:
        os.makedirs(node_k.node_path, exist_ok=True)

    trainer_raw_split = raw_split(args, dataloader_cluster_k, node_k, epochs=args.epochs_raw_split, 
                                  noise_start= args.noise_start, sample_interval = args.sample_interval,
                                  collapse_check_loss=collapse_check_loss)

    #if save_node_path is not None:
    #    np.save(save_node_path, node_k)  

    return node_k

def check_stop_refinement_condition(node, text_logs_path, min_prob_mass_variation = 150):
    if len(node.child_left.cluster_probs)>=3:
        prob_mass_variation = (node.child_left.cluster_probs[-1].numpy() - node.child_left.cluster_probs[-2].numpy())
        prob_mass_variation = np.abs(prob_mass_variation).sum()
        log_headings = get_log_heading("CHECKING CONDITION FOR CONTINUING REFINEMENTS FOR NODE {}".format(node.name), spacing=2)
        print_save_log('\n\n\n' + log_headings, text_logs_path)  
        print_save_log("Condition for continuing refinements: total prob mass variation between the last 2 refinements must be > {}.".format(min_prob_mass_variation), text_logs_path)
        print_save_log("(As a heuristic to save up time, we assume negligible variation indicates a clustering local minimum unlikely to change with more refinements)", text_logs_path)
        print_save_log('The variation of prob. mass for the last 2 refinemnets is: {:.2f}.'.format(prob_mass_variation), text_logs_path)
        if prob_mass_variation < min_prob_mass_variation:
            print_save_log('Canceling next refinements for this node.', text_logs_path)
            return True
        else:
            print_save_log('Continuing next refinements for this node. ', text_logs_path)
            return False
    else:
        return False
                        
def refine_tree_nodes(args, node_k, dataloader_train, ith_refinement, no_refinements, halt_epoch = 20, collapse_check_loss=0.01, save_node_path=None):

    ith_refinement = len(node_k.child_left.cluster_probs)
    
    dataloader_cluster_l = copy.deepcopy(dataloader_train)
    dataloader_cluster_m = copy.deepcopy(dataloader_train)
    dataloader_cluster_l.sampler.weights = node_k.child_left.cluster_probs[-1]
    dataloader_cluster_m.sampler.weights = node_k.child_right.cluster_probs[-1]

    trainer_ref= refinement(args, dataloader_cluster_l, dataloader_cluster_m, epochs=args.epochs_refinement, 
                            noise_start= args.noise_start, ref_it=ith_refinement,
                            sample_interval=args.sample_interval, collapse_check_loss=collapse_check_loss,
                            node_k = node_k, print_vars=False)

    dataloader_cluster_l.sampler.weights = node_k.child_left.cluster_probs[-1]
    dataloader_cluster_m.sampler.weights = node_k.child_right.cluster_probs[-1]

    #if save_node_path is not None:
    #    np.save(save_node_path, node_k)  
    
    return node_k

def grow_tree_from_root(root_node, dataloader_train, args):
    os.makedirs(args.logs_path, exist_ok=True)
    text_logs_path = os.path.join(args.logs_path, "global_tree_logs.txt")
    save_log_text('', text_logs_path, open_mode='w')
    for i in range(args.no_splits):     
        split_node = search_node_to_split(root_node, text_logs_path=text_logs_path)        
        split_node = raw_split_tree_node(args, split_node, dataloader_train, save_node_path='root_node')
        non_leaf_list = get_non_leaf_nodes_list(root_node)
        leaf_list = get_leaf_nodes_list(root_node)
        log_title_raw_split = 'GLOBAL TREE LOGS AFTER RAW SPLIT OF NODE {}'.format(split_node.name)
        view_global_tree_logs(dataloader_train, non_leaf_list, leaf_list, text_logs_path, log_title=log_title_raw_split) 
        for j in range(args.no_refinements):
            stop_refinement_flag = check_stop_refinement_condition(split_node, text_logs_path, args.min_prob_mass_variation)
            if not(stop_refinement_flag):
                split_node = refine_tree_nodes(args, split_node, dataloader_train, ith_refinement=j, no_refinements=args.no_refinements, save_node_path='root_node')
                if j == args.no_refinements-1:
                    log_headings = get_log_heading("END OF RIFENEMENTS FOR NODE {} SPLIT".format(split_node.name), spacing=2)
                    print_save_log("\n\n\n" + log_headings, text_logs_path)
                    print_save_log("{}/{} refinements concluded.".format(args.no_refinements, args.no_refinements), text_logs_path)
                non_leaf_list = get_non_leaf_nodes_list(root_node)
                leaf_list = get_leaf_nodes_list(root_node)
                log_title_ref = 'GLOBAL TREE LOGS AFTER REFINEMENT {} OF NODE {} SPLIT'.format(j+1, split_node.name)
                view_global_tree_logs(dataloader_train, non_leaf_list, leaf_list, text_logs_path, log_title=log_title_ref)
            else: 
                split_node.child_left.status += ", skipped {}".format(args.no_refinements-j)
                split_node.child_right.status += ", skipped {}".format(args.no_refinements-j)
                #np.save('root_node', root_node)  
                log_headings = get_log_heading("END OF RIFINEMENTS FOR NODE {} SPLIT".format(split_node.name), spacing=2)
                print_save_log("\n\n\n" + log_headings, text_logs_path)
                print_save_log("{}/{} refinements concluded. Remaining {} refinements skipped due to negligible variation.".format(
                                j,  args.no_refinements,  args.no_refinements-j), text_logs_path)
                break

    
    
