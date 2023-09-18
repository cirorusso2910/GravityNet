# PARAMETERS

| **EXECUTION MODE**             | **DESCRIPTION**                                                 |
|--------------------------------|-----------------------------------------------------------------|
| **train**                      | train model                                                     |
| **resume**                     | resume training from a specific epoch                           |
| **test**                       | test model                                                      |
| **test_NMS**                   | test model with Non-Maxima-Suppression (NMS) post-processing    |
| **train_test**                 | train and test model                                            |

### INITIALIZATION
| **PARAMETER** | **DESCRIPTION**    |
|---------------|--------------------|
| **where**     | where to save/read |

### LOAD DATASET
| **PARAMETER**    | **DESCRIPTION**  |
|------------------|------------------|
| **dataset**      | dataset name     |
| **image_height** | image height (H) |
| **image_width**  | image width (W)  |
| **split**        | dataset split    |

### EXPERIMENT ID
| **PARAMETER** | **DESCRIPTION**    |
|---------------|--------------------|
| **typeID**    | experiment ID type |

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

| **PARAMETER**           | **DESCRIPTION**                               |
|-------------------------|-----------------------------------------------|
| **rescale**             | image rescale factor                          |
| **max_padding**         | padding size for annotation                   |

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
| **PARAMETER**       | **DESCRIPTION**                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| **epochs**          | number of epochs                                                                                              |
| **epoch_to_resume** | number of epoch to resume                                                                                     |
| **optimizer**       | Optimizer                                                                                                     |
| **scheduler**       | Scheduler                                                                                                     |
| **clip_gradient**   | Clip Gradient                                                                                                 |
| **learning_rate**   | how fast approach the minimum                                                                                 |
| **lr_patience**     | number of epochs with no improvement after which learning rate will be reduced [scheduler: ReduceLROnPlateau] |
| **lr_step_size**    | how much the learning rate decreases [scheduler: StepLR]                                                      |
| **lr_gamma**        | multiplicative factor of learning rate decay [scheduler: StepLR]                                              |
| **max_norm**        | max norm of the gradients to be clipped [Clip Gradient]                                                       |

### GRAVITY LOSS
| **PARAMETER** | **DESCRIPTION**                                                       |
|---------------|-----------------------------------------------------------------------|
| **alpha**     | alpha parameter for loss                                              |
| **gamma**     | gamma parameter for loss                                              |
| **lambda**    | lambda factor for loss sum                                            |
| **hook**      | hook distance                                                         |
| **gap**       | hook gap distance in classification loss for rejection gravity points |

### EVALUATION
| **PARAMETER**   | **DESCRIPTION**                                      |
|-----------------|------------------------------------------------------|
| **eval**        | evaluation criterion                                 |
| **FP_images**   | type of images on which calculate FP                 |
| **work_point**  | average FP for scan to get sensitivity in FROC Curve |

### LOAD MODEL
| **PARAMETER**                   | **DESCRIPTION**                             |
|---------------------------------|---------------------------------------------|
| **load_best_sensitivity_model** | load best model with sensitivity work point |
| **load_best_AUFROC_model**      | load best model with AUFROC [0, 10]         |

### OUTPUT
| **PARAMETER**         | **DESCRIPTION**            |
|-----------------------|----------------------------|
| **type_draw**         | type output draw           |
| **box_draw_radius**   | box radius to draw         |
| **do_output_gravity** | do output gravity          |
| **num_image**         | num images to show in test |
| **idx**               | index image in dataset     |

### OUTPUT FPS
| **PARAMETER** | **DESCRIPTION** |
|---------------|-----------------|
| **FPS**       | FPS             |

### POST PROCESSING
| **PARAMETER**      | **DESCRIPTION**                                                                  |
|--------------------|----------------------------------------------------------------------------------|
| **do_NMS**         | apply Non-Maxima-Suppression (NMS)                                               |
| **NMS_box_radius** | Non-Maxima-Suppression (NMS) box radius for gravity points->boxes transformation |

### ROCalc
| **PARAMETER**       | **DESCRIPTION**                |
|---------------------|--------------------------------|
| **type_detections** | type of detections file (.csv) |

### PLOT CHECK
| **PARAMETER**           | **DESCRIPTION**                            |
|-------------------------|--------------------------------------------|
| **plot_check_list**     | list parameters for plot-check             |
| **type_plot_check**     | type of plot-check                         |
| **do_plots_train**      | do plots-train check and save results      |
| **do_plots_validation** | do plots-validation check and save results |
| **do_plots_test**       | do plots-test check and save results       |
| **do_plots_test_NMS**   | do plots-test-NMS check and save results   |
| **do_plots_test_all**   | do plots-test-all check and save results   |
| **do_metrics**          | do metrics check and save results          |
| **do_plots**            | do plots check and save results            |

### DEBUG
| **PARAMETER**                     | **DESCRIPTION**                                   |
|-----------------------------------|---------------------------------------------------|
| **debug_execution**               | stop execution before starting execution mode     |
| **debug_initialization**          | no experiment results creation                    |
| **debug_transforms**              | stop execution in dataset transforms              |
| **debug_transforms_augmentation** | stop execution in dataset augmentation transforms |
| **debug_anchors**                 | save gravity points configuration                 |
| **debug_hooking**                 | save gravity points hooking                       |
| **debug_network**                 | save network summary model                        |
| **debug_test**                    | show first detections during test                 |
| **debug_validation**              | show first detections during validation           |
| **debug_FROC**                    | show FROC computation debug                       |
