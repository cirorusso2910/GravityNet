# Parameter reference

For the shortest setup path, follow the [README quick start](../../README.markdown#quick-start). Use this page when you want to customize an experiment.

Run `python GravityNet.py train_test --help` from the repository root for CLI flags and choices. Settings resolve in this order: **CLI flags > YAML file > built-in defaults**. The [default settings](../../net/parameters/parameters_default.py) define YAML keys; the [parser](../../net/parameters/parameters.py) defines CLI spellings and aliases.

```bash
python GravityNet.py train_test --cfg configs/INbreast.yaml --epochs 50
```

YAML uses names without `--`, such as `epochs: 50` and `batch_size_train: 1`. Use `lambda` in YAML and `--lambda_factor` on the CLI. Unknown YAML keys are rejected. Keep your input YAML: the generated experiment `config.yaml` contains extra snapshot fields and cannot be passed directly to `--cfg`.

**NOTE**: every boolean parameter below (`pretrained`, `do_dataset_augmentation`, `save_config`,
`clip_gradient`, `load_best_sensitivity_10_FPS_model`, `load_best_AUFROC_0_10_model`,
`load_best_AUPR_model`, `do_output_gravity`, `do_NMS`) uses `argparse.BooleanOptionalAction`:
pass the flag on its own to enable it, or `--no-<flag>` to force it off (also overrides a default
of `True`, e.g. `--no-clip_gradient`).

### CONFIGURATION
| **PARAMETER** | **DESCRIPTION**                                                                          |
|----------------|-------------------------------------------------------------------------------------------|
| **cfg**        | path to a YAML file overriding parameter defaults (CLI flags still take precedence over it) |

| **EXECUTION MODE** | **DESCRIPTION**                       |
|--------------------|---------------------------------------|
| **train**          | train model                           |
| **test**           | test model                            |
| **train_test**     | train and test model                  |
|                    |                                       |
| **explainability** | explainability mode                   |
|                    |                                       |
| **script_anchors** | script-anchors execution mode         |
| **script_dataset** | script-dataset execution mode         |

### INITIALIZATION
| **PARAMETER**        | **DESCRIPTION** |
|----------------------|-----------------|
| **dataset_path**     | parent of the dataset folder; reads `<dataset_path>/<dataset>/`    |
| **experiments_path** | output root; create it before training and use a separate root for independent runs |

### LOAD DATASET
| **PARAMETER**    | **DESCRIPTION**  |
|------------------|------------------|
| **dataset**      | dataset name     |
| **small_lesion** | small lesion     |
| **image_height** | image height (H) |
| **image_width**  | image width (W)  |
| **split**        | selects `split/split-<value>.csv`; Cervix93 uses `default`    |

## UTILITY DATASET
| **PARAMETER**              | **DESCRIPTION**       |
|----------------------------|-----------------------|
| **images_extension**        | image extension       |
| **images_masks_extension** | image mask extension  |
| **annotations_extension**  | annotations extension |

### EXPERIMENT ID
| **PARAMETER** | **DESCRIPTION**          |
|---------------|--------------------------|
| **typeID**    | experiment ID type       |
| **sep**       | experiment ID separator (`_` by default; use the original separator when testing an older run)  |

### DEVICE
| **PARAMETER**   | **DESCRIPTION**   |
|-----------------|-------------------|
| **GPU**         | label included in the experiment ID; use CUDA_VISIBLE_DEVICES to select visible GPUs   |
| **num_threads** | number of threads |

### REPRODUCIBILITY
| **PARAMETER** | **DESCRIPTION**          |
|---------------|--------------------------|
| **seed**      | seed for reproducibility |

### DATASET NORMALIZATION
| **PARAMETER** | **DESCRIPTION**       |
|---------------|-----------------------|
| **norm**      | dataset normalization |

### DATASET TRANSFORMS

| **PARAMETER**           | **DESCRIPTION**             |
|-------------------------|-----------------------------|
| **rescale**             | image rescale factor        |
| **num_channels**        | number of image channels    |
| **max_padding**         | padding size for annotation |

### DATASET AUGMENTATION
| **PARAMETER**               | **DESCRIPTION**         |
|-----------------------------|-------------------------|
| **do_dataset_augmentation** | do dataset augmentation |

### DATA LOADER
| **PARAMETER**        | **DESCRIPTION**                                                                                         |
|----------------------|---------------------------------------------------------------------------------------------------------|
| **batch_size_train** | batch size for train                                                                                    |
| **batch_size_val**   | batch size for validation                                                                               |
| **batch_size_test**  | batch size for test                                                                                     |
| **num_workers**      | numbers of sub-processes to use for data loading <br/> if 0 the data will be loaded in the main process |

### NETWORK
| **PARAMETER**  | **DESCRIPTION**  |
|----------------|------------------|
| **backbone**   | Backbone model   |
| **pretrained** | initialize the backbone with pretrained weights (may download them) |

### GRAVITY POINTS
| **PARAMETER**  | **DESCRIPTION**              |
|----------------|------------------------------|
| **config**     | gravity points configuration |

### HYPER-PARAMETERS
| **PARAMETER**     | **DESCRIPTION**                                                                                               |
|-------------------|---------------------------------------------------------------------------------------------------------------|
| **epochs**        | number of epochs                                                                                              |
|                   |                                                                                                               |
| **optimizer**     | Optimizer                                                                                                     |
| **scheduler**     | Scheduler                                                                                                     |
| **clip_gradient** | Clip Gradient                                                                                                 |
|                   |                                                                                                               |
| **learning_rate** | how fast approach the minimum                                                                                 |
| **lr_patience**   | number of epochs with no improvement after which learning rate will be reduced [scheduler: ReduceLROnPlateau] |
| **lr_step_size**  | how much the learning rate decreases [scheduler: StepLR]                                                      |
| **lr_gamma**      | multiplicative factor of learning rate decay [scheduler: StepLR]                                              |
| **lr_T0**         | number of epochs for the first restart [scheduler: CosineAnnealing]                                           |
|                   |                                                                                                               |
| **max_norm**      | max norm of the gradients to be clipped [Clip Gradient]                                                       |

### GRAVITY LOSS
| **PARAMETER** | **DESCRIPTION**                                                       |
|---------------|-----------------------------------------------------------------------|
| **alpha**     | alpha parameter for loss                                              |
| **gamma**     | gamma parameter for loss                                              |
| **lambda_factor** | loss weight (YAML key: `lambda`)                                            |
| **hook**      | hook distance                                                         |
| **gap**       | hook gap distance in classification loss for rejection gravity points |

### EVALUATION
| **PARAMETER**       | **DESCRIPTION**                      |
|---------------------|--------------------------------------|
| **eval**            | evaluation criterion                 |
| **FP_images**       | type of images on which calculate FP |
| **score_threshold** | score threshold                      |

### LOAD MODEL
| **PARAMETER**                          | **DESCRIPTION**                         |
|----------------------------------------|-----------------------------------------|
| **load_best_sensitivity_10_FPS_model** | load best model with sensitivity 10 FPS |
| **load_best_AUFROC_0_10_model**        | load best model with AUFROC [0, 10]     |
| **load_best_AUPR_model**               | load best model with AUPR               |

### OUTPUT
| **PARAMETER**         | **DESCRIPTION**            |
|-----------------------|----------------------------|
| **type_draw**         | type output draw           |
| **box_draw_radius**   | box radius to draw         |
| **do_output_gravity** | do output gravity          |
| **num_images**         | num images to show in test |
| **idx**               | index image in dataset     |

### POST PROCESSING
| **PARAMETER**      | **DESCRIPTION**                                                                  |
|--------------------|----------------------------------------------------------------------------------|
| **do_NMS**         | apply Non-Maxima-Suppression (NMS)                                               |
| **NMS_box_radius** | Non-Maxima-Suppression (NMS) box radius for gravity points->boxes transformation |
