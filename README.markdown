![alt text](doc/logo/GravityNet-logo.png)

## :information_source: INFO

**GravityNet** official repository <br>
by **PhD Student Ing. Ciro Russo** <br>
:man_technologist: [LinkedIn](https://www.linkedin.com/in/ciro-russo-b14056100/) <br>
:bookmark_tabs: [Google Scholar](https://scholar.google.com/citations?user=XJV2vVsAAAAJ&hl=it&authuser=1) <br>
:books: [Research Gate](https://www.researchgate.net/profile/Ciro-Russo-4/research) <br>
:computer: [Kaggle](https://www.kaggle.com/cirorusso2910) <br>

:office: **University of Cassino and Lazio Meridionale** <br>
:man_teacher: Prof. **Claudio Marrocco** ([Google Scholar](https://scholar.google.it/citations?user=ed4B7I4AAAAJ&hl=it)) <br>
:man_teacher: Prof. **Alessandro Bria** ([LinkedIn](https://www.linkedin.com/in/alessandro-bria-831ab149/?originalSubdomain=it))<br>
:man_technologist: **Giulio Russo** ([LinkedIn](https://www.linkedin.com/in/russogiulio/)) <br>

----------------------------------------------------------------------
## :rocket: GRAVITY NET

**GravityNet** is novel **one-stage** **end-to-end** **detector** specifically designed to **detect** **small** **lesions** in **medical** **images**.
Precise localization of small lesions presents challenges due to their appearance and the diverse contextual backgrounds in which they are found.
To address this, our approach introduces a **new type of pixel-based anchor** that dynamically moves towards the targeted lesion for detection.
We refer to this new architecture as **GravityNet**, and the novel anchors as **gravity points** since they appear to be “attracted” by the lesions.

<img src="doc/architecture/GravityNet-architecture.png" width="80%" height="80%" alt="">

----------------------------------------------------------------------
## :page_facing_up: REFERENCE
**Paper**: [GravityNet for end-to-end small lesion detection](https://www.sciencedirect.com/science/article/abs/pii/S0933365724000848#preview-section-snippets) <br>

    @article{Russo_Bria_Marrocco_2024, <br>
             title =   {GravityNet for end-to-end small lesion detection}, <br>
             ISSN={0933-3657}, <br>
             DOI={10.1016/j.artmed.2024.102842}, <br>
             journal={Artificial Intelligence in Medicine}, <br>
             author={Russo, Ciro and Bria, Alessandro and Marrocco, Claudio}, <br>
             year={2024}, <br>
             month=mar, <br>
             pages={102842} <br>
    }

**ArXiv**: https://arxiv.org/abs/2309.12876

----------------------------------------------------------------------
## :robot: HOW TO GRAVITY

<!-- :construction: UNDERGOING MAINTENANCE :construction: -->

### :one: PARAMETERS
This framework uses parameters-parsing, so each **new** parameter **must** be added paying attention to the reference section 
(for details see [parameters](doc/code/parameters.markdown)).

The definition of these parameters is essential for the _experiment_ID_ to save the results and avoid overwriting.    
**NOTE**: for Windows users use a _separator_ equal to ' _ ', while for Linux users the default separator is ' | '.

For details about the [requirements](doc/requirements/requirements.markdown)

    !pip install -r requirements.txt

----------------------------------------------------------------------
### :two: INITIALIZATION
Before starting the experiment, it is necessary to define the working paths:

    --dataset_path -> path where the dataset is located
    --experiments_path -> path where to save the result of the experiment

**NOTE**: the _dataset_path_ is concatenated to the dataset name (see [parameters](doc/code/parameters.markdown))

For details about the [dataset-structure](./datasets/dataset-structure.markdown) <br>
For details about the [experiments-structure](./doc/experiments/experiments-structure.markdown) <br>

----------------------------------------------------------------------
### :three: CLASS DATASET
The **Class Dataset** is defined according to the structure of the dataset (see [dataset-structure](./datasets/dataset-structure.markdown)). 

**NOTE**: run **Dataset-Statistics.py** (in _script-dataset_) to save the dataset statistics.

| SCRIPT-DATASET        | DESCRIPTION             |
|-----------------------|-------------------------|
| Dataset-Statistics.py | Save dataset statistics |

----------------------------------------------------------------------
### :four: SPLIT DATA
To split the data into the **train**, **validation**, and **test** subsets,
the framework uses a **split** file defined in the dataset folder.

The splits used in the experiments are [reported](datasets).

| DATASET     | SMALL LESION        | REFERENCE                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|-------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| INbreast    | microcalcifications | [INbreast reference](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiAoL7NgK2BAxXE0wIHHWurDDMQFnoECBQQAQ&url=https%3A%2F%2Fwww.sciencedirect.com%2Fscience%2Farticle%2Fabs%2Fpii%2FS107663321100451X&usg=AOvVaw1r-qXP0Rk4qGao1LfKkqCc&opi=89978449)                                                                                                                                                      |
| E-ophtha-MA | microaneurysms      | [E-ophtha-MA reference](https://www.sciencedirect.com/user/identity/landing?code=Um_NMyFZ6dAD9fJwYGT9iOtLbjcoF1g8f48bRZ-G&state=retryCounter%3D0%26csrfToken%3D23a2ff6e-a0a8-42a5-ae5d-b904009ac4d4%26idpPolicy%3Durn%253Acom%253Aelsevier%253Aidp%253Apolicy%253Aproduct%253Ainst_assoc%26returnUrl%3D%252Fscience%252Farticle%252Fpii%252FS1959031813000237%253Fvia%25253Dihub%26prompt%3Dnone%26cid%3Darp-f12057f3-3362-4f06-9758-826d42268be4) |
| Cervix93    | nuclei              | [Cervix93 reference](https://github.com/parham-ap/cytology_dataset)                                                                                                                                                                                                                                                                                                                                                                                |

----------------------------------------------------------------------
### :five: DATASET INFORMATION
All information about the dataset are reported in the **statistics** of the dataset

----------------------------------------------------------------------
### :six: DATASET TRANSFORMS
Transformations on each **sample** in the dataset are defined by a **Class**.

The framework uses a **collate function** to define the transformations to be applied,
depending on the normalization type: **none**, **min-max**, and **std**.

The **transformations** to be applied vary depending on the application and the type of data used, 
to this end, we provide basic transformations. 

**Augmentation transformations** can be applied to the train dataset, as: **Horizontal**, **Vertical Flipping**

----------------------------------------------------------------------
### :seven: GRAVITY-POINTS CONFIGURATION
To see the **gravity-points configuration** and the **hooking process** in _script-anchors_ are provided the codes

| SCRIPT-DATASET                  | DESCRIPTION                         |
|---------------------------------|-------------------------------------|
| Gravity-Points-Configuration.py | Save gravity-points configuration   |
| Gravity-Points-Hooking.py       | Save gravity-points hooking process |

----------------------------------------------------------------------
### :eight: GRAVITY NET ARCHITECTURE
GravityNet is a **one-stage** **end-to-end** **detector** composed of a backbone network and two specific subnetworks. <br>
The **backbone** is a convolutional network and plays the role of feature extractor. <br>
The **first subnet** performs convolutional object classification on the backbone's output. <br>
The **second subnet** performs convolutional gravity-points regression.

The available **backbone**:

| ResNet     | ResNeXt           | DenseNet     | EfficientNet    | EfficientNetV2   | SwinTransformer | 
|------------|-------------------|--------------|-----------------|------------------|-----------------|
| ResNet-18  | ResNeXt-50_32x4d  | DenseNet-121 | EfficientNet-B0 | EfficientNetV2-S | Swin-T          |
| ResNet-34  | ResNeXt-101_32x8d | DenseNet-161 | EfficientNet-B1 | EfficientNetV2-M | Swin-S          |
| ResNet-50  | ResNeXt-101_64x4d | DenseNet-169 | EfficientNet-B2 | EfficientNetV2-L | Swin-B          |
| ResNet-101 |                   | DenseNet-201 | EfficientNet-B3 |                  |                 |
| ResNet-152 |                   |              | EfficientNet-B4 |                  |                 |
|            |                   |              | EfficientNet-B5 |                  |                 |
|            |                   |              | EfficientNet-B6 |                  |                 |
|            |                   |              | EfficientNet-B7 |                  |                 |


----------------------------------------------------------------------
### :nine: EXECUTION MODE
The available **execution mode**:

| EXECUTION MODE | DESCRIPTION                                                  |
|----------------|--------------------------------------------------------------|
| train          | train model                                                  |
| test           | test model                                                   |
| train_test     | train and test model                                         |

----------------------------------------------------------------------
### :keycap_ten: EXAMPLE OF EXECUTION

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u GravityNet.py train_test 
        --dataset_path              =       "path to dataset main folder"
        --experiments_path          =       "path to experiments result"
        --images_extension          =       [png, tif]
        --annotations_extension     =       csv
        --dataset                   =       "Dataset Name"
        --do_dataset_augmentation
        --num_channels              =       [1, 3]
        --small_lesion              =       [microcalcifications, microaneurysms, nuclei]
        --split                     =       [default, 1-fold, 2-fold]
        --rescale                   =       [0,5, 1.0]
        --norm                      =       [none, min-max, std] 
        --epochs                    =       100
        --lr                        =       1e-04
        --bs                        =       8
        --backbone                  =       ResNet-152
        --pretrained
        --config                    =       grid-15
        --hook                      =       15
        --eval                      =       distance10
        --FP_images                 =       normal
        --score_threshold           =       0.05
        --do_NMS                    
        --NMS_box_radius            =       "Lesion Radius"
        --do_output_gravity
        --num_images                =       "Num Images"

----------------------------------------------------------------------
