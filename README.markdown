# GRAVITY NET

## ARCHITECTURE
GravityNet is a **one-stage** **end-to-end** **detector** composed of a backbone network and two specific subnetworks. <br>
The **backbone** is a convolutional network and plays the role of feature extractor. <br>
The **first subnet** performs convolutional object classification on the backbone's output; <br> 
the **second subnet** performs convolutional gravity-points regression.

<img src="doc/architecture/GravityNet-architecture.png" width="80%" height="80%" alt="">

## EXECUTION INSTRUCTIONS
The following instructions must be followed to properly run GravityNet:

1. Initialization
2. Dataset requirements
3. Parameters-parsing
4. Execution

### INITIALIZATION

The structure, by which the data is organized, is to be defined in:

    /net/initialization/init/folders/dataset_name_folders.py 

in form of a dictionary.
        
    example:
    
    dataset_folders = {
        'images': 'images',
        'masks': 'masks',
        'annotations': 'annotations'
    }

Having defined the structure, the construction of the data paths (_annotations_, _images_ and _masks_) takes place in:
    
    /net/initialization/init.py

and are stored in a specific dictionary: _path_dataset_dict_

### DATASET REQUIREMENTS

Dataset class, to read data samples, is to be defined in
    
    /net/dataset/DatasetName.py

the annotation header can be defined in _/net/initialization/header/annotations.py_

A split file **must** be defined to divide the data into the subsets of _train_, _validation_ and _test_. <br>
The splits used in the experiments are shown in _/datasets/_ according to the dataset.

Modifications for reading information from the dataset will follow from here on <br>
(alternatively, they may not be applied and will be calculated at each run):

    /net/dataset/dataset_num_images                 -> to modify num images according to dataset split
    /net/dataset/dataset_num_normal_images          -> to modify num normal images according to dataset split
    /net/dataset/dataset_transforms                 -> to modify dataset transforms according to dataset split
    /net/dataset/dataset_transforms_augmentation    -> to modify dataset transforms for augmentation according to dataset split

### PARAMETERS-PARSING

Parameter-parsing is handled in _net/parameters_ <br>
parameters are also defined [here](doc/code/parameters.markdown)

### EXECUTION

Example of command for execution:
    
    # with E-ophtha-MA
    CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py train --where=home --dataset=E-ophtha-MA --do_dataset_augmentation --split=1-fold --norm=min-max --channel=G --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-152 --pretrained --config=grid-10 --hook=10 --eval=radius1 --do_output_gravity > train-example-1fold.txt
    
    # with INbreast
    CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py train --where=data --dataset=INbreast --do_dataset_augmentation --split=1-fold --rescale=1.0 --norm=none --epochs=1 --lr=1e-04 --bs=8 --backbone=ResNet-34 --pretrained --config=grid-10 --hook=10 --eval=distance7 --do_output_gravity > train-example-1fold.txt










