# GRAVITY NET

**GravityNet** repository

**Paper**: [link]()

## EXECUTION INSTRUCTIONS
The following instructions must be followed to properly run the GravityNet:

1. Parameters-parsing
2. Initialization
3. Class Dataset
4. Split data
5. Dataset information
6. Dataset transforms
7. Gravity points configuration
8. Parameters summary
9. Execution mode
10. Example of execution

### 1. PARAMETERS-PARSING
This application uses parameters-parsing, so each **new** parameter **must** be added paying attention to the reference section <br>
(for details see [parameters](doc/code/parameters.markdown)).

    Parameters is defined in:
        net/parameters

The definition of these parameters is essential for the building of an _experiment_ID_ to save results and avoid overwriting.
    
    Experiment ID is defined in:
        net/initialization/ID/experiment_ID.py

----------------------------------------------------------------------

### 2. INITIALIZATION
Before any modification to the source implementation, it is necessary to define work paths.
    
    the $WHERE$ parameter is defined to determine which $PATH$ definition

    $PATH$ is defined in:
        net/initialization/folders/default_folders.py
    according to $WHERE$ parameter to manage multiple work paths

    $DATASET-STRUCTURE$ is defined in:
        net/initialization/folders/dataset_folders.py
    The dataset-structure is defined in the form of a dictionary (an example is given in the code)

After defining the working paths, the _dict_ must be concatenated to obtain the correct paths
    
    all path is defined in
        net/initialization/init.py

For details about the [dataset-structure](./datasets/dataset-structure.markdown) <br>
For details about the [experiments-structure](./doc/experiments/experiments-structure.markdown) <br>

----------------------------------------------------------------------

### 3. CLASS DATASET
The **Class Dataset** must be defined according to the dataset-structure 
and the _$FILE_EXTENSION$_ for each data type..

**Hint:** rename the dataset.py with the name of the dataset $DATASET$

    The Class Dataset is defined in:
        net/dataset/dataset.py

Below a table with tha data types used:

| **DATA TYPE** | **$FILE_EXTENSION$** |
|---------------|----------------------|
| IMAGE         | tif, jpg             |
| IMAGE MASK    | png                  |
| ANNOTATION    | csv                  |

For ease of use, there is the option of managing the annotation header with a function:

    Annotation header is defined in:
        net/initialization/header/annotations.py

----------------------------------------------------------------------

### 4. SPLIT DATA
To split the data into _train_, _validation_ and _test_ subsets a **split** file is used. <br>
It is mandatory to define a **split-$N$-fold.csv** defined in the _split_ dataset subfolder.

It is important (unless many changes are made later) to maintain the following format for the split file. <br>
        
    Header: INDEX, FILENAME, SPLIT
    - INDEX: sequential element index (0 to N-1), where N is the total number of samples
    - FILENAME: filename of the sample
    - SPLIT: split type (choices of: train, validation and test)

The splits used in the experiments on the 
[INbreast](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiAoL7NgK2BAxXE0wIHHWurDDMQFnoECBQQAQ&url=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fabs%2Fpii%2FS107663321100451X&usg=AOvVaw1r-qXP0Rk4qGao1LfKkqCc&opi=89978449)
and
[E-ophtha-MA](https://www.sciencedirect.com/user/identity/landing?code=Um_NMyFZ6dAD9fJwYGT9iOtLbjcoF1g8f48bRZ-G&state=retryCounter%3D0%26csrfToken%3D23a2ff6e-a0a8-42a5-ae5d-b904009ac4d4%26idpPolicy%3Durn%253Acom%253Aelsevier%253Aidp%253Apolicy%253Aproduct%253Ainst_assoc%26returnUrl%3D%252Fscience%252Farticle%252Fpii%252FS1959031813000237%253Fvia%25253Dihub%26prompt%3Dnone%26cid%3Darp-f12057f3-3362-4f06-9758-826d42268be4)
are [reported](datasets) <br>
**NOTE**: 2-fold image-based cross-validation was performed for each dataset

----------------------------------------------------------------------

### 5. DATASET INFORMATION
All information about the $DATASET$ must be added
(to avoid extra computational costs at each execution) for each split used.

    $DATASET$ num images
        net/dataset/dataset_num_images.py

    $DATASET$ num normal images
        net/dataset/dataset_num_normal_images.py

    $DATASET$ num annotations
        net/dataset/num_annotations.py

Optionally, functions are available:
    
    num normal images: compute the number of normal images (images without lesion) for each subset
        net/dataset/statistics/num_normal_images.py

    num annotations images: compute the number of lesions for each subset
        net/dataset/statistics/num_annotations_images.py

----------------------------------------------------------------------

### 6. DATASET TRANSFORMS
The transformations on each sample in the dataset is handled by a specific function 
(such as pre-processing or normalization).

    $DATASET$ transforms
        net/dataset/dataset_transforms

Likewise for transformations of data augmentation

    $DATASET$ augmentation transforms
        net/dataset/dataset_augmentation

By default, there are three types of normalization: **none**, **min-max** and **std**. <br>
All data transformations are defined in a specific path and defined with a Class.

    $DATASET$ data transforms
        net/dataset/transforms

    $DATASET$ data augmentation transforms
        net/dataset/transforms_augmentation

Some example transforms are given in code:

- **Basic data transforms**: <br>
    AnnotationPadding.py <br>
    Add3ChannelsImage.py <br>
    MinMaxNormalization.py <br>
    Rescale.py <br>
    Resize.py <br>
    SelectImageChannel.py <br>
    SelectMaxRadius.py <br>
    StandardNormalization.py <br>
    ToTensor.py <br>


- **Augmentation transforms**: <br>
    MyHorizontalAndVerticalFlip.py <br>
    MyHorizontalFlip.py <br>
    MyVerticalFlip.py <br>
  
**NOTE**: MinMaxNormalization.py and StandardNormalization.py read the statistics 
from _statistics_ folder of dataset
(in alternatively they can be entered manually in the code in the corresponding dictionary fields).

----------------------------------------------------------------------

### 7. GRAVITY POINTS CONFIGURATION
For the correct generation of the gravity point configuration,
the dimensions of the image **must** be specified: height (H) and width (W).

**! IMPORTANT !**: it **must** be taken into account if you perform operations 
such as _cropping_ or _resize/rescale_ with the image size set.

----------------------------------------------------------------------

### 8. PARAMETERS SUMMARY
It should be modified as required, showing an overview of all parameters 
and information on the $DATASET$ used.

    parameters summary
        net/parameters/parameters_summary.py

----------------------------------------------------------------------

### 9. EXECUTION MODE
Below the available **execution mode**:

| EXECUTION MODE | DESCRIPTION                                                  |
|----------------|--------------------------------------------------------------|
| train          | train model                                                  |
| resume         | resume training from a specific epoch                        |
| test           | test model                                                   |
| test_NMS       | test model with Non-Maxima-Suppression (NMS) post-processing |
| train_test     | train and test model                                         |

----------------------------------------------------------------------

### 10. EXAMPLE OF EXECUTION

    CUDA_VISIBLE_DEVICES=3 python3 -u GravityNet.py $EXECUTION_MODE$ --$PARAMETERS$
