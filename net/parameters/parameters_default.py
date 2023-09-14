parameters_default = {

    # -------------- #
    # INITIALIZATION #
    # -------------- #
    'where': 'home',

    # ------------ #
    # LOAD DATASET #
    # ------------ #
    'dataset': 'INbreast',
    'image_height': 3328,
    'image_width': 2560,
    'split': '1-fold',

    # ------------- #
    # EXPERIMENT ID #
    # ------------- #
    'typeID': 'default',

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
    # INbreast (GravityNet-microcalcifications)
    'orientation': 'L',
    'image_height_crop': 3328,
    'image_width_crop': 2560,

    # E-ophtha-MA (GravityNet-microaneurysms)
    'image_height_resize': 1216,
    'image_width_resize': 1408,
    'resize_tool': 'PyTorch',
    'channel': 'RGB',

    # common
    'rescale': 1.0,
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
    'work_point': 10,

    # ---------- #
    # LOAD MODEL #
    # ---------- #
    'load_best_sensitivity_model': True,
    'load_best_AUFROC_model': False,

    # ------ #
    # OUTPUT #
    # ------ #
    'type_draw': 'box',
    'do_output_gravity': False,
    'num_images': 1,
    'idx': 0,

    # ---------- #
    # OUTPUT FPS #
    # ---------- #
    'FPS': 10,

    # --------------- #
    # POST PROCESSING #
    # --------------- #
    'do_NMS': False,
    'NMS_box_radius': 1,

    # ------ #
    # ROCalc #
    # ------ #
    'type_detections': 'test',

    # ---------- #
    # PLOT CHECK #
    # ---------- #
    'plot_check_list': "",
    'type_plot_check': "",
    'do_plots_train': False,
    'do_plots_validation': False,
    'do_plots_test': False,
    'do_plots_test_NMS': False,
    'do_plots_test_all': False,
    'do_metrics': False,
    'do_plots': False,

    # ----- #
    # DEBUG #
    # ----- #
    'debug_execution': False,
    'debug_initialization': False,
    'debug_transforms': False,
    'debug_transforms_augmentation': False,
    'debug_anchors': False,
    'debug_hooking': False,
    'debug_network': False,
    'debug_test': False,
    'debug_validation': False,
    'debug_FROC': False,

}
