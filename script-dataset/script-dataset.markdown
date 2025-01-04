# SCRIPT-DATASET

## :bar_chart: Dataset-Statistics.py

Save dataset statistics

    Dataset-Statistics.py script_dataset
        --dataset_path              =       "path to dataset main folder"
        --experiments_path          =       "path to experiments result"
        --images_extension          =       [png, tif]
        --images_masks_extension    =       [png, none]
        --annotations_extension     =       csv
        --dataset                   =       "Dataset Name"
        --small_lesion              =       [microcalcifications, microaneurysms, nuclei]
        --split                     =       [default, 1-fold, 2-fold]
        --rescale                   =       [0,5, 1.0]
        --num_channels              =       [1, 3]

| DATASET     | STATISTICS                                                   |
|-------------|--------------------------------------------------------------|
| INbreast    | [INbreast statistics](../datasets/INbreast/statistics)       |
| E-ophtha-MA | [E-ophtha-MA statistics](../datasets/E-ophtha-MA/statistics) |
| Cervix93    | [Cervix93 statistics](../datasets/Cervix93/statistics)       |
