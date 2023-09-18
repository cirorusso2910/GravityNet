from net.parameters.parameters_default import parameters_default

parameters_help = {

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    'mode': "EXECUTION MODE",
    'train': 'train model',
    'test': 'test model',
    'test_NMS': 'test model with Non-Maxima-Suppression (NMS) post-processing',
    'train_test': 'train and test model',

    'who_is_my_creator': 'who is my creator?',

    # -------------- #
    # INITIALIZATION #
    # -------------- #
    'where': f"initialization path for results (default: '{parameters_default['where']}')",

    # ------------ #
    # LOAD DATASET #
    # ------------ #
    'dataset': f"dataset name (default: '{parameters_default['dataset']}')",
    'image_height': f"image height size (default: {parameters_default['image_height']}",
    'image_width': f"image width size (default: {parameters_default['image_width']}",
    'split': f"dataset split (default: '{parameters_default['split']}')",

    # ------------- #
    # EXPERIMENT ID #
    # ------------- #
    'typeID': f"experiment ID type (default: {parameters_default['typeID']}",

    # ------ #
    # DEVICE #
    # ------ #
    'GPU': f"GPU device name (default: {parameters_default['GPU']}",
    'num_threads': f"number of threads (default: {parameters_default['num_threads']}",

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #
    'seed': f"seed for reproducibility (default: {parameters_default['seed']})",

    # --------------------- #
    # DATASET NORMALIZATION #
    # --------------------- #
    'norm': f"dataset normalization (default: {parameters_default['norm']}",

    # ------------------ #
    # DATASET TRANSFORMS #
    # ------------------ #
    'rescale': f"image rescale factor (default: {parameters_default['rescale']})",
    'max_padding': f"padding size for annotation (default: {parameters_default['max_padding']})",

    # -------------------- #
    # DATASET AUGMENTATION #
    # -------------------- #
    'do_dataset_augmentation': f"do dataset augmentation (default: {parameters_default['do_dataset_augmentation']}",

    # ----------- #
    # DATA LOADER #
    # ----------- #
    'batch_size_train': f"batch size for train (default: {parameters_default['batch_size_train']})",
    'batch_size_val': f"batch size for validation (default: {parameters_default['batch_size_val']})",
    'batch_size_test': f"batch size for test (default: {parameters_default['batch_size_test']})",

    'num_workers': f"numbers of sub-processes to use for data loading, if 0 the data will be loaded in the main process (default: {parameters_default['num_workers']})",

    # ------- #
    # NETWORK #
    # ------- #
    'backbone': f"Backbone model (default: {parameters_default['backbone']})",
    'pretrained': f"PreTrained model (default: {parameters_default['pretrained']})",

    # -------------- #
    # GRAVITY POINTS #
    # -------------- #
    'config': f"gravity points configuration (default: {parameters_default['config']})",

    # ---------------- #
    # HYPER-PARAMETERS #
    # ---------------- #
    'epochs': f"number of epochs (default: {parameters_default['epochs']})",
    'epoch_to_resume': f"number of epoch to resume (default: {parameters_default['epoch_to_resume']})",

    'optimizer': f"Optimizer (default: '{parameters_default['optimizer']}'",
    'scheduler': f"Scheduler (default: '{parameters_default['scheduler']}'",
    'clip_gradient': f"Clip Gradient (default: '{parameters_default['clip_gradient']}'",

    'learning_rate': f"how fast approach the minimum (default: {parameters_default['learning_rate']})",
    'lr_momentum': f"momentum factor [optimizer: SGD] (default: {parameters_default['lr_momentum']})",
    'lr_patience': f"number of epochs with no improvement after which learning rate will be reduced [scheduler: ReduceLROnPlateau] (default: {parameters_default['lr_patience']})",
    'lr_step_size': f"how much the learning rate decreases [scheduler: StepLR] (default: {parameters_default['lr_step_size']})",
    'lr_gamma': f"multiplicative factor of learning rate decay [scheduler: StepLR] (default: {parameters_default['lr_gamma']})",

    'max_norm': f"max norm of the gradients to be clipped [Clip Gradient] (default: {parameters_default['max_norm']})",

    # ------------ #
    # GRAVITY LOSS #
    # ------------ #
    'alpha': f"alpha parameter for loss (default: {parameters_default['alpha']}",
    'gamma': f"gamma parameter for loss (default: {parameters_default['gamma']}",
    'lambda': f"lambda factor for loss sum (default: {parameters_default['lambda']}",
    'hook': f"hook distance (default: {parameters_default['hook']}",
    'gap': f"hook gap distance in classification loss for rejection gravity points (default: {parameters_default['gap']})",

    # ---------- #
    # EVALUATION #
    # ---------- #
    'eval': f"evaluation criterion (default: '{parameters_default['eval']}'",
    'FP_images': f"type of images on which calculate FP (default: '{parameters_default['FP_images']}'",
    'work_point': f"average FP for scan to get sensitivity in FROC Curve (default: {parameters_default['work_point']}",

    # ---------- #
    # LOAD MODEL #
    # ---------- #
    'load_best_sensitivity_model': f"load best model with sensitivity work point (default: '{parameters_default['load_best_sensitivity_model']}'",
    'load_best_AUFROC_model': f"load best model with AUFROC [0, 10] (default: '{parameters_default['load_best_AUFROC_model']}'",

    # ------ #
    # OUTPUT #
    # ------ #
    'type_draw': f"type output draw (default: {parameters_default['type_draw']}",
    'box_draw_radius': f"box radius to draw (default: {parameters_default['box_draw_radius']}",
    'do_output_gravity': f"do output gravity (default: {parameters_default['do_output_gravity']}",
    'num_images': f"num images to show in test (default: {parameters_default['num_images']}",
    'idx': f"index image in dataset (default: {parameters_default['idx']}",

    # ---------- #
    # OUTPUT FPS #
    # ---------- #
    'FPS': f"FPS (default: {parameters_default['FPS']}",

    # --------------- #
    # POST PROCESSING #
    # --------------- #
    'do_NMS': f"apply Non-Maxima-Suppression (NMS) (default: {parameters_default['do_NMS']}",
    'NMS_box_radius': f"Non-Maxima-Suppression (NMS) box radius for gravity points->boxes transformation (default: {parameters_default['NMS_box_radius']}",

    # ------ #
    # ROCalc #
    # ------ #
    'type_detections': f"type of detections file (.csv) (default: {parameters_default['type_detections']}",

    # ---------- #
    # PLOT CHECK #
    # ---------- #
    'plot_check_list': f"list parameters for plot check (default: {parameters_default['plot_check_list']}",
    'type_plot_check': f"type plot check (default: {parameters_default['type_plot_check']}",
    'do_plots_train': f"do plots train (default: {parameters_default['do_plots_train']}",
    'do_plots_validation': f"do plots validation (default: {parameters_default['do_plots_validation']}",
    'do_plots_test': f"do plots test (default: {parameters_default['do_plots_test']}",
    'do_plots_test_NMS': f"do plots test NMS (default: {parameters_default['do_plots_test_NMS']}",
    'do_plots_test_all': f"do plots test all (default: {parameters_default['do_plots_test_all']}",
    'do_metrics': f"do metrics (default: {parameters_default['do_metrics']}",
    'do_plots': f"do plots (default: {parameters_default['do_plots']}",

    # ----- #
    # DEBUG #
    # ----- #
    'debug_execution': f"stop execution before starting execution mode (default: {parameters_default['debug_execution']}",
    'debug_initialization': f"no experiment results creation (default: {parameters_default['debug_initialization']}",
    'debug_transforms': f"stop execution in dataset transforms (default: {parameters_default['debug_transforms']}",
    'debug_transforms_augmentation': f"stop execution in dataset augmentation transforms (default: {parameters_default['debug_transforms_augmentation']}",
    'debug_anchors': f"save gravity points configuration (default: {parameters_default['debug_anchors']}",
    'debug_hooking': f"save gravity points hooking (default: {parameters_default['debug_hooking']}",
    'debug_network': f"save network summary model (default: {parameters_default['debug_network']}",
    'debug_test': f"show first detections during test (default: {parameters_default['debug_test']}",
    'debug_validation': f"show first detections during validation (default: {parameters_default['debug_validation']}",
    'debug_FROC': f"show FROC computation debug (default: {parameters_default['debug_FROC']}",
}
