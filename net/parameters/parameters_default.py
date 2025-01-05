parameters_default = {

    # -------------- #
    # INITIALIZATION #
    # -------------- #
    'dataset_path': 'datasets',
    'experiments_path': 'experiments',

    # ------------ #
    # LOAD DATASET #
    # ------------ #
    'dataset': 'INbreast',
    'small_lesion': 'calcifications',
    'image_height': 3328,
    'image_width': 2560,
    'split': '1-fold',

    # --------------- #
    # UTILITY DATASET #
    # --------------- #
    'images_extension': 'tiff',
    'images_masks_extension': 'png',
    'annotations_extension': 'csv',

    # ------------- #
    # EXPERIMENT ID #
    # ------------- #
    'typeID': 'default',
    'sep': '|',

    # ------ #
    # DEVICE #
    # ------ #
    'GPU': 'None',
    'num_threads': 32,

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #
    'seed': 0,

    # --------------------- #
    # DATASET NORMALIZATION #
    # --------------------- #
    'norm': 'none',

    # ------------------ #
    # DATASET TRANSFORMS #
    # ------------------ #
    'rescale': 1.0,
    'num_channels': 1,
    'max_padding': 200,

    # -------------------- #
    # DATASET AUGMENTATION #
    # -------------------- #
    'do_dataset_augmentation': False,

    # ----------- #
    # DATA LOADER #
    # ----------- #
    'batch_size_train': 8,
    'batch_size_val': 8,
    'batch_size_test': 8,

    'num_workers': 8,

    # ------- #
    # NETWORK #
    # ------- #
    'backbone': 'ResNet-18',
    'pretrained': False,

    # -------------- #
    # GRAVITY POINTS #
    # -------------- #
    'config': 'grid-10',
    'save_config': False,

    # ---------------- #
    # HYPER-PARAMETERS #
    # ---------------- #
    'epochs': 1,
    'epoch_to_resume': 0,

    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'clip_gradient': True,

    'learning_rate': 1e-4,
    'lr_momentum': 0.1,  # optimizer: SGD
    'lr_patience': 3,  # scheduler: ReduceLROnPlateau
    'lr_step_size': 3,  # scheduler: StepLR
    'lr_gamma': 0.1,  # scheduler: StepLR

    'max_norm': 0.1,

    # ------------ #
    # GRAVITY LOSS #
    # ------------ #
    'alpha': 0.25,
    'gamma': 2.0,
    'lambda': 10,

    'hook': 10,
    'gap': 0,

    # ---------- #
    # EVALUATION #
    # ---------- #
    'eval': 'distance7',
    'FP_images': 'normals',
    'score_threshold': 0.0,

    # ---------- #
    # LOAD MODEL #
    # ---------- #
    'load_best_sensitivity_10_FPS_model': True,
    'load_best_AUFROC_0_10_model': False,
    'load_best_AUPR_model': False,

    # ------ #
    # OUTPUT #
    # ------ #
    'type_draw': 'box',
    'box_draw_radius': 10,
    'do_output_gravity': False,
    'num_images': 1,
    'idx': 0,

    # --------------- #
    # POST PROCESSING #
    # --------------- #
    'do_NMS': False,
    'NMS_box_radius': 1,

}
