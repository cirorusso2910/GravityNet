# SCRIPT-DATASET

**NOTE**: run from the repository root with `PYTHONPATH=.` so the `net` package resolves.

## 📊 Dataset-Statistics.py

Save dataset statistics

    PYTHONPATH=. python3 script-dataset/Dataset-Statistics.py script_dataset \
        --dataset_path "path to dataset main folder" \
        --experiments_path "path to experiments result" \
        --images_extension tiff \
        --images_masks_extension png \
        --annotations_extension csv \
        --dataset "Dataset Name" \
        --small_lesion microcalcifications \
        --split 1-fold \
        --rescale 1.0 \
        --num_channels 1

| DATASET     | STATISTICS                                                  |
|-------------|--------------------------------------------------------------|
| INbreast    | [INbreast statistics](../datasets/INbreast/statistics)       |
| E-ophtha-MA | [E-ophtha-MA statistics](../datasets/E-ophtha-MA/statistics) |
| Cervix93    | [Cervix93 statistics](../datasets/Cervix93/statistics)       |
