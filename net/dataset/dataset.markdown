# NET.DATASET
**net.dataset** package

## STRUCTURE

    dataset
    |
    | - draw
    |   | add_3_channels_image.py
    |   | draw_annotation.py
    |   | draw_cut.py
    |   | select_1_channel_image.py
    |
    | - statistics
    |   | min_max_statistics.py
    |   | num_annotations.py
    |   | num_normal_images.py
    |   | standard_statistics.py
    |
    | - transforms
    |   -> transforms
    |
    | - transforms_augmentation
    |   -> transforms-augmentation
    |
    | - utility
    |   | read_dataset_sample.py
    |   | split_index.py
    |   | viewable_image.py
    |
    | dataset.py
    | dataset_augmentation.py
    | dataset_num_annotations.py
    | dataset_num_images.py
    | dataset_num_normal_images.py
    | dataset_split.py
    | dataset_statistics.py
    | dataset_transforms
    | dataset_transforms_augmentation.py

## DOCUMENTATION

| FOLDER                      | FUNCTION                           | DESCRIPTION                                                            |
|-----------------------------|------------------------------------|------------------------------------------------------------------------|
| **draw**                    | add_3_channels_image.py            | Add 3 channels to image: copy image 3 times                            |
| **draw**                    | draw_annotation.py                 | Draw annotation on image and save                                      |
| **draw**                    | draw_cut.py                        | Draw cut line on image and save                                        |
| **draw**                    | select_1_channel_image.py          | Add 3 channels to image: copy image 3 times                            |
| **statistics**              | min_max_statistics.py              | Compute (min, max) value of dataset                                    |
| **statistics**              | num_annotations.py                 | Get num annotations                                                    |
| **statistics**              | num_normal_images.py               | Get num normal images                                                  |
| **statistics**              | standard_statistics.py             | Compute (mean, std) value of dataset                                   |
| **statistics**              | standard_statistics.py             | Compute (mean, std) value of dataset                                   |
| **transforms**              |                                    | transforms                                                             |
| **transforms_augmentation** |                                    | transforms-augmentation                                                |
| **utility**                 | read_dataset_sample.py             | Read dataset sample at specific index (idx) position                   |
| **utility**                 | split_index.py                     | Get split index                                                        |
| **utility**                 | viewable_image.py                  | Transforms the input image in a 'viewable' image in the range [0, 255] |
|                             | dataset.py                         | Dataset Class                                                          |
|                             | dataset_augmentation.py            | Apply dataset augmentation (only for dataset-train)                    |
|                             | dataset_num_annotations.py         | Compute dataset num annotations                                        |
|                             | dataset_num_images.py              | Compute dataset num images                                             |
|                             | dataset_num_normal_images.py       | Compute dataset num normal images                                      |
|                             | dataset_split.py                   | Compute dataset num normal images                                      |
|                             | dataset_statistics.py              | Compute dataset statistics: (min, max) and (mean, std)                 |
|                             | dataset_transforms.py              | Collect dataset transforms                                             |
|                             | dataset_transforms_augmentation.py | Collect dataset transforms augmentation (only for dataset-train)       |
