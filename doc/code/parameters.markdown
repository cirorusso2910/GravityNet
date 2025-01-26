# PARAMETERS

| **EXECUTION MODE** | **DESCRIPTION**                       |
|--------------------|---------------------------------------|
| **train**          | train model                           |
| **resume**         | resume training from a specific epoch |
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
| **dataset_path**     | dataset path    |
| **experiments_path** | experiment path |

### LOAD DATASET
| **PARAMETER**    | **DESCRIPTION**  |
|------------------|------------------|
| **dataset**      | dataset name     |
| **small_lesion** | small lesion     |
| **image_height** | image height (H) |
| **image_width**  | image width (W)  |
| **split**        | dataset split    |

## UTILITY DATASET
| **PARAMETER**              | **DESCRIPTION**       |
|----------------------------|-----------------------|
| **image_extension**        | image extension       |
| **images_masks_extension** | image mask extension  |
| **annotations_extension**  | annotations extension |

### EXPERIMENT ID
| **PARAMETER** | **DESCRIPTION**          |
|---------------|--------------------------|
| **typeID**    | experiment ID type       |
| **sep**       | separator experiment ID  |

### DEVICE
| **PARAMETER**   | **DESCRIPTION**   |
|-----------------|-------------------|
| **GPU**         | GPU device name   |
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
| **pretrained** | PreTrained model |

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
|                   |                                                                                                               |
| **max_norm**      | max norm of the gradients to be clipped [Clip Gradient]                                                       |

### GRAVITY LOSS
| **PARAMETER** | **DESCRIPTION**                                                       |
|---------------|-----------------------------------------------------------------------|
| **alpha**     | alpha parameter for loss                                              |
| **gamma**     | gamma parameter for loss                                              |
| **lambda**    | lambda factor for loss sum                                            |
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
| **num_image**         | num images to show in test |
| **idx**               | index image in dataset     |

### POST PROCESSING
| **PARAMETER**      | **DESCRIPTION**                                                                  |
|--------------------|----------------------------------------------------------------------------------|
| **do_NMS**         | apply Non-Maxima-Suppression (NMS)                                               |
| **NMS_box_radius** | Non-Maxima-Suppression (NMS) box radius for gravity points->boxes transformation |
