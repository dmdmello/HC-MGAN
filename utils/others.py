import os
import math
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
import torch
from models.gan import GAN

def sum_dicts(dict_a, dict_b):
    assert(dict_a.keys() == dict_b.keys())
    return {k:dict_a[k]+dict_b[k] for k,v in dict_a.items()}

def zero_dict_values(dict):
    return {k:0 for k,v in dict.items()}

def save_log_text(log_text, save_path, open_mode = 'a'):
    try:
        with open(save_path, open_mode) as f:
            f.write(log_text)
    except FileNotFoundError:
        print("Path {} for saving training logs does not exist".format(save_path))

def print_save_log(log, save_path, print_log=True):
    if print_log:
        print(log)
    if save_path is not None: 
        save_log_text(remove_bold_from_string(str(log))+'\n', save_path, open_mode='a')

def get_log_heading(text, spacing=0):
    hyphen_bar =  (len(text)+2)*'-' 
    line_break = ('#\n')*spacing
    return get_bold_string(hyphen_bar + '\n'+ line_break  + '# ' + text + '\n'+ line_break + hyphen_bar)

def get_bold_string(string):
    return "\033[1m" + string + "\033[0m"

def remove_bold_from_string(string):
    string = string.replace('\033[1m', '')
    string = string.replace('\033[0m', '')
    return string

def create_gans(args, no_gans=1, no_g_paths=2):
    
    available_img_dim = [28, 32]

    import models.models_general as mdg
    if args.img_dim == 32:
        import models.models_32x32 as md
    elif args.img_dim == 28:
        import models.models_28x28 as md
    else:
        raise  ValueError('Data type {} not available, choose from {}'.format(args.data_type, available_img_dim))
    
    def create_feature_layer():
        return md.EncoderLayers(architecture=args.architecture_d,
                                nf = args.nf_d, 
                                kernel_size=args.kernel_size_d, 
                                norm=args.normalization_d,
                                nc=args.img_channels,
                                print_shapes=True)

    gan_list = []

    def create_gen():
        return md.Generator(architecture = args.architecture_g, 
                            nf = args.nf_g, 
                            kernel_size = args.kernel_size_g, 
                            latent_dim = args.latent_dim, 
                            nc = args.img_channels,
                            norm = args.normalization_g,
                            print_shapes=True)
    
    if args.shared_features_across_ref:
        shared_feature_layers = create_feature_layer().cuda()

    for i in range(no_gans):

        gen = create_gen().cuda()
        
        gens = [gen]
        for i in range(no_g_paths-1):
            gens.append(create_gen().cuda())

        gen_set = mdg.GeneratorSet(*gens)

        if args.shared_features_across_ref:
            feature_layers = shared_feature_layers
        else:
            feature_layers = create_feature_layer().cuda()

        disc = mdg.Discriminator(feature_layers).cuda()
        clasf = mdg.Classifier(feature_layers, no_c_outputs=2).cuda()
            
        #optimizers
        optimizer_G = torch.optim.Adam(list(gen_set.parameters()), lr=args.lr_g, betas=(args.b1, args.b2))
        optimizer_D = torch.optim.Adam(list(disc.parameters()), lr=args.lr_d, betas=(args.b1, args.b2))
        optimizer_C = torch.optim.Adam(list(clasf.linear_clasf.parameters()), lr=args.lr_c, betas=(args.b1, args.b2))

        gan = GAN(gen_set, disc, clasf, feature_layers, optimizer_G, optimizer_D, optimizer_C, args.diversity_parameter_g)

        gan_list.append(gan)
        
    return gan_list

'''def get_pretty_df(df):

    dfStyler = df.style.set_properties(**{'text-align': 'center', 
                                          'border' : '1px  solid !important' })
    df = dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    return df'''