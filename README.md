# HC-MGAN
Code for the paper "Top-Down Deep Clustering with Multi-generator GANs"


## Running the experiments with python (.py) files

Reproducing experiments on python is straightforward. Provided the datasets are obtained (if not, see "Getting the Datasets" section), just run each .py with the respective dataset name, using default arguments 
(with the only required argument being the dataset folder where the main folder of all required datasets are located).

To reproduce FMNIST:

$ python fmnist.py --dataset_path=path/to/data

To reproduce MNIST :

$ python mnist.py --dataset_path=path/to/data

To reproduce SOP :

$ python sop.py --dataset_path=path/to/data

If you wish to select different arguments, use --help to view other options.


## Running the experiments on jupyter notebook (.ipynb) files

Our results are much better visualized in jupyter notebooks (the clustering tables printed by our program in particular look a lot better). 

We recommend using the jupyter version of each experiment we provided, they are: sop.ipynb for SOP, mnist.ipynb for MNIST and fmnist.ipynb for FMNIST. 

We can't use arguments in jupyter notebooks, so the same arguments used for the .py experiments are replaced by variables in each experiment's notebook.

Each variable related to each argument needs to be manually edited in order to have their default value replaced. 

The original values for each variable are the ones used in the experiment, like the default argument values on the .py experiment files.

YOU STILL NEED TO PROVIDE A PATH TO THE DATASET in the dataset_path variable inside each notebook. 


## Experiment Saved Logs

Both .py and notebook experiments save their logs in a folder specified as --logs_path argument for .py experiments and as the logs_path variable in the notebooks.

If the folder does not exist, it is created by the program. If it exists, each log value will be subscribed by the new experiment using the same path for log saving.

Everything the program prints on the terminal or on the jupyter notebook gets saved as a log. 

The logs pertaining to the tree growth are stored in path/to/logs/global_tree_logs.txt.

Logs related to each individual node training are stored in a subfolder dedicated to that node in path/to/logs/X, where X is the node name assigned by the program. 

Raw Split training logs of node X are stored at a subfolder inside X given by path/to/logs/X/raw_split

The i-th refinement training logs of node X are stored at a subfolder inside X given by path/to/logs/X/refinement_i.

GAN/MGAN images generated throughout each node's refinement or raw split training can't be printed on the terminal (they are printed only on jupyter notebooks), but are saved in the raw split or refinement subfolders for each node.


## GPU

The code assumes an available GPU, and the default device is set to 0. 

To use another device whose index is X, specify as an argument --device=X for the .py experiments or change the "device" variable value to X in the notebook experiments.

It's encouraged to activate AMP (automatic mixed precision) when running each experiment, as it makes the code run a lot faster on modern GPUs.  

To activate AMP with the .py experiments, provide --amp_enable as an argument, for example:

$ python fmnist.py --dataset_path=path/to/data --amp-enable

To activate AMP on the notebook, set the "device" boolean variable to true.

## Getting the Datasets

### SOP
SOP has to be downloaded manually at http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip.
Additionally, the images of SOP need to be resized to 32x32 resolution, and the classes "kettle" and "lamp" need to be removed as mentioned in the paper. 
The following linux commands are sufficient to obtain this dataset as we used in our experiments (for resizing we use the "mogrify" command from "ImageMagick", so you need to have ImageMagick installed):

$ wget 'http://ftp.cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
$ unzip Stanford_Online_Products.zip -d path/to/data/
$ cd path/to/data/Stanford_Online_Products
$ rm -r kettle_final
$ rm -r lamp_final
$ find . -name '*.JPG' -execdir mogrify -resize 32x32! {} +

When running the experiment, you need to specify path/to/data as a value for the argument --dataset-path, that is, the path where the SOP root folder named exactly "Stanford_Online_Products" is located. 
DO NOT USE path/to/data/Stanford_Online_Products as a value for --dataset-path or SOP won't be found. 
MAKE SURE the SOP root folder is NAMED EXACTLY as "Stanford_Online_Products", or it won't be found either. 

Optionally, you might rename each class folder inside "Stanford_Online_Products" to remove the "_final" suffix, so that the tables and logs printed by our program look a lot cleaner:

$ mv bicycle_final bicycle
$ mv cabinet_final cabinet
$ mv chair_final chair
$ mv coffee_maker_final coffee_maker
$ mv fan_final fan
$ mv mug_final mug
$ mv sofa_final sofa
$ mv stapler_final stapler
$ mv table_final table
$ mv toaster_final toaster


### MNIST and FMNIST 

If you have already downloaded either MNIST or FMNIST with pytorch, just use the path to the folder where the root folder of each dataset is located.
If you haven't downloaded either, you can use any folder as --dataset-path and the program will automatically download MNIST or FMNIST to this path and read them from there. 


## Requirements

This code was executed using pytorch 1.9 and 1.10, and it requires a GPU (as mentioned earlier, you can use the --device argument when running each experiment to set a specific device).

Main required packages are pandas, matplotlib, tqdm, scikit-learn, scipy, torchvision, but a few more might be required. 
We recommend installing everything with miniconda, and what can't be find with miniconda, installed with pip.

Results are much better visualized in jupyter notebooks (jupyter notebooks for the three experiments sop.ipynb, mnist.ipynb, fmnist.ipynb are provided).

## Problems with the Code

Contact me immediatelly (dani-dmello@hotmail.com) if you have any trouble running the code. It's likely that unpredictable issues will be present during the first few days we realease the code before we have time to fix them. 
