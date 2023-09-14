import argparse

from net.parameters.parameters_choices import parameters_choices
from net.parameters.parameters_default import parameters_default
from net.parameters.parameters_help import parameters_help


def parameters_parsing() -> argparse.Namespace:
    """
    Definition of parameters-parsing for each execution mode

    :return: parser of parameters parsing
    """

    # parser
    parser = argparse.ArgumentParser(description='Argument Parser')

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    parser_mode = parser.add_subparsers(title=parameters_help['mode'], dest='mode', metavar='mode')

    # execution mode
    parser_train = parser_mode.add_parser('train', help=parameters_help['train'])
    parser_resume = parser_mode.add_parser('resume', help=parameters_help['resume'])
    parser_test = parser_mode.add_parser('test', help=parameters_help['test'])
    parser_test_NMS = parser_mode.add_parser('test_NMS', help=parameters_help['test_NMS'])
    parser_train_test = parser_mode.add_parser('train_test', help=parameters_help['train_test'])

    parser_output_FPS = parser_mode.add_parser('output_FPS', help=parameters_help['output_FPS'])
    parser_sensitivity_FPS = parser_mode.add_parser('sensitivity_FPS', help=parameters_help['sensitivity_FPS'])

    # execution mode script
    parser_script_anchors = parser_mode.add_parser('script_anchors', help=parameters_help['script_anchors'])
    parser_script_dataset = parser_mode.add_parser('script_dataset', help=parameters_help['script_dataset'])
    parser_script_debug = parser_mode.add_parser('script_debug', help=parameters_help['script_debug'])
    parser_script_detections = parser_mode.add_parser('script_detections', help=parameters_help['script_detections'])
    parser_script_metrics = parser_mode.add_parser('script_metrics', help=parameters_help['script_metrics'])
    parser_script_model = parser_mode.add_parser('script_model', help=parameters_help['script_model'])
    parser_script_output = parser_mode.add_parser('script_output', help=parameters_help['script_output'])
    parser_script_output_paper = parser_mode.add_parser('script_output_paper', help=parameters_help['script_output_paper'])
    parser_script_plot = parser_mode.add_parser('script_plot', help=parameters_help['script_plot'])
    parser_script_plot_check = parser_mode.add_parser('script_plot_check', help=parameters_help['script_plot_check'])
    parser_script_plot_check_complete = parser_mode.add_parser('script_plot_check_complete', help=parameters_help['script_plot_check_complete'])
    parser_script_plot_check_paper = parser_mode.add_parser('script_plot_check_paper', help=parameters_help['script_plot_check_paper'])
    parser_script_test = parser_mode.add_parser('script_test', help=parameters_help['script_test'])
    parser_script_test_complete = parser_mode.add_parser('script_test_complete', help=parameters_help['script_test_complete'])
    parser_script_time = parser_mode.add_parser('script_time', help=parameters_help['script_time'])
    parser_script_utility = parser_mode.add_parser('script_utility', help=parameters_help['script_utility'])
    parser_script_vs = parser_mode.add_parser('script_vs', help=parameters_help['script_vs'])

    # who is my creator
    parser_who_is_my_creator = parser_mode.add_parser('who_is_my_creator', help=parameters_help['who_is_my_creator'])

    # execution mode list
    execution_mode = [parser_train,
                      parser_resume,
                      parser_test,
                      parser_test_NMS,
                      parser_train_test,
                      parser_output_FPS,
                      parser_sensitivity_FPS,
                      parser_script_anchors,
                      parser_script_dataset,
                      parser_script_debug,
                      parser_script_detections,
                      parser_script_metrics,
                      parser_script_model,
                      parser_script_output,
                      parser_script_output_paper,
                      parser_script_plot,
                      parser_script_plot_check,
                      parser_script_plot_check_complete,
                      parser_script_plot_check_paper,
                      parser_script_test,
                      parser_script_test_complete,
                      parser_script_time,
                      parser_script_utility,
                      parser_script_vs,
                      parser_who_is_my_creator]

    # for each subparser 'mode'
    for subparser in execution_mode:

        # -------------- #
        # INITIALIZATION #
        # -------------- #
        subparser.add_argument('--where',
                               type=str,
                               default=parameters_default['where'],
                               help=parameters_help['where'])

        # ------------ #
        # LOAD DATASET #
        # ------------ #
        subparser.add_argument('--dataset',
                               type=str,
                               default=parameters_default['dataset'],
                               help=parameters_help['dataset'])

        subparser.add_argument('--image_height',
                               type=int,
                               default=parameters_default['image_height'],
                               help=parameters_help['image_height'])

        subparser.add_argument('--image_width',
                               type=int,
                               default=parameters_default['image_width'],
                               help=parameters_help['image_width'])

        subparser.add_argument('--split',
                               type=str,
                               default=parameters_default['split'],
                               help=parameters_help['split'])

        # ------------- #
        # EXPERIMENT ID #
        # ------------- #
        subparser.add_argument('--typeID',
                               type=str,
                               default=parameters_default['typeID'],
                               help=parameters_help['typeID'])

        # ------ #
        # DEVICE #
        # ------ #
        subparser.add_argument('--GPU',
                               type=str,
                               default=parameters_default['GPU'],
                               help=parameters_help['GPU'])

        subparser.add_argument('--num_threads',
                               type=int,
                               default=parameters_default['num_threads'],
                               help=parameters_help['num_threads'])

        # --------------- #
        # REPRODUCIBILITY #
        # --------------- #
        subparser.add_argument('--seed',
                               type=int,
                               default=parameters_default['seed'],
                               help=parameters_help['seed'])

        # --------------------- #
        # DATASET NORMALIZATION #
        # --------------------- #
        subparser.add_argument('--norm',
                               type=str,
                               default=parameters_default['norm'],
                               help=parameters_help['norm'])

        # ------------------ #
        # DATASET TRANSFORMS #
        # ------------------ #
        # INbreast (GravityNet-microcalcifications)
        subparser.add_argument('--orientation',
                               type=str,
                               default=parameters_default['orientation'],
                               help=parameters_help['orientation'])

        subparser.add_argument('--image_height_crop',
                               type=int,
                               default=parameters_default['image_height_crop'],
                               help=parameters_help['image_height_crop'])

        subparser.add_argument('--image_width_crop',
                               type=int,
                               default=parameters_default['image_width_crop'],
                               help=parameters_help['image_width_crop'])

        # E-ophtha-MA (GravityNet-microaneurysms)
        subparser.add_argument('--image_height_resize',
                               type=int,
                               default=parameters_default['image_height_resize'],
                               help=parameters_help['image_height_resize'])

        subparser.add_argument('--image_width_resize',
                               type=int,
                               default=parameters_default['image_width_resize'],
                               help=parameters_help['image_width_resize'])

        subparser.add_argument('--resize_tool',
                               type=str,
                               default=parameters_default['resize_tool'],
                               help=parameters_help['resize_tool'])

        subparser.add_argument('--channel',
                               type=str,
                               default=parameters_default['channel'],
                               help=parameters_help['channel'])

        # common
        subparser.add_argument('--rescale',
                               type=float,
                               default=parameters_default['rescale'],
                               help=parameters_help['rescale'])

        subparser.add_argument('--max_padding',
                               type=int,
                               default=parameters_default['max_padding'],
                               help=parameters_help['max_padding'])

        # -------------------- #
        # DATASET AUGMENTATION #
        # -------------------- #
        subparser.add_argument('--do_dataset_augmentation',
                               action='store_true',
                               default=parameters_default['do_dataset_augmentation'],
                               help=parameters_help['do_dataset_augmentation'])

        # ----------- #
        # DATA LOADER #
        # ----------- #
        subparser.add_argument('--batch_size_train', '--bs',
                               type=int,
                               default=parameters_default['batch_size_train'],
                               help=parameters_help['batch_size_train'])

        subparser.add_argument('--batch_size_val',
                               type=int,
                               default=parameters_default['batch_size_val'],
                               help=parameters_help['batch_size_val'])

        subparser.add_argument('--batch_size_test',
                               type=int,
                               default=parameters_default['batch_size_test'],
                               help=parameters_help['batch_size_test'])

        subparser.add_argument('--num_workers',
                               type=int,
                               default=parameters_default['num_workers'],
                               help=parameters_help['num_workers'])

        # ------- #
        # NETWORK #
        # ------- #
        subparser.add_argument('--backbone',
                               type=str,
                               default=parameters_default['backbone'],
                               choices=parameters_choices['backbone'],
                               help=parameters_help['backbone'])

        subparser.add_argument('--pretrained',
                               action='store_true',
                               default=parameters_default['pretrained'],
                               help=parameters_help['pretrained'])

        # -------------- #
        # GRAVITY POINTS #
        # -------------- #
        subparser.add_argument('--config',
                               type=str,
                               default=parameters_default['config'],
                               help=parameters_help['config'])

        # ---------------- #
        # HYPER-PARAMETERS #
        # ---------------- #
        subparser.add_argument('--epochs', '--ep',
                               type=int,
                               default=parameters_default['epochs'],
                               help=parameters_help['epochs'])

        subparser.add_argument('--epoch_to_resume', '--ep_to_resume',
                               type=int,
                               default=parameters_default['epoch_to_resume'],
                               help=parameters_help['epoch_to_resume'])

        subparser.add_argument('--optimizer',
                               type=str,
                               default=parameters_default['optimizer'],
                               choices=parameters_choices['optimizer'],
                               help=parameters_help['optimizer'])

        subparser.add_argument('--scheduler',
                               type=str,
                               default=parameters_default['scheduler'],
                               choices=parameters_choices['scheduler'],
                               help=parameters_help['scheduler'])

        subparser.add_argument('--clip_gradient',
                               action='store_true',
                               default=parameters_default['clip_gradient'],
                               help=parameters_help['clip_gradient'])

        subparser.add_argument('--learning_rate', '--lr',
                               type=float,
                               default=parameters_default['learning_rate'],
                               help=parameters_help['learning_rate'])

        subparser.add_argument('--lr_momentum',
                               type=int,
                               default=parameters_default['lr_momentum'],
                               help=parameters_help['lr_momentum'])

        subparser.add_argument('--lr_patience',
                               type=int,
                               default=parameters_default['lr_patience'],
                               help=parameters_help['lr_patience'])

        subparser.add_argument('--lr_step_size',
                               type=int,
                               default=parameters_default['lr_step_size'],
                               help=parameters_help['lr_step_size'])

        subparser.add_argument('--lr_gamma',
                               type=int,
                               default=parameters_default['lr_gamma'],
                               help=parameters_help['lr_gamma'])

        subparser.add_argument('--max_norm',
                               type=float,
                               default=parameters_default['max_norm'],
                               help=parameters_help['max_norm'])

        # ------------ #
        # GRAVITY LOSS #
        # ------------ #
        subparser.add_argument('--alpha',
                               type=float,
                               default=parameters_default['alpha'],
                               help=parameters_help['alpha'])

        subparser.add_argument('--gamma',
                               type=float,
                               default=parameters_default['gamma'],
                               help=parameters_help['gamma'])

        subparser.add_argument('--lambda_factor',
                               type=int,
                               default=parameters_default['lambda'],
                               help=parameters_help['lambda'])

        subparser.add_argument('--hook',
                               type=int,
                               default=parameters_default['hook'],
                               help=parameters_help['hook'])

        subparser.add_argument('--gap',
                               type=int,
                               default=parameters_default['gap'],
                               help=parameters_help['gap'])

        # ---------- #
        # EVALUATION #
        # ---------- #
        subparser.add_argument('--eval',
                               type=str,
                               default=parameters_default['eval'],
                               help=parameters_help['eval'])

        subparser.add_argument('--FP_images',
                               type=str,
                               default=parameters_default['FP_images'],
                               help=parameters_help['FP_images'])

        subparser.add_argument('--work_point', '--wp',
                               type=int,
                               default=parameters_default['work_point'],
                               help=parameters_help['work_point'])

        # ---------- #
        # LOAD MODEL #
        # ---------- #
        subparser.add_argument('--load_best_sensitivity_model',
                               action='store_true',
                               default=parameters_default['load_best_sensitivity_model'],
                               help=parameters_help['load_best_sensitivity_model'])

        subparser.add_argument('--load_best_AUFROC_model',
                               action='store_true',
                               default=parameters_default['load_best_AUFROC_model'],
                               help=parameters_help['load_best_AUFROC_model'])

        # ------ #
        # OUTPUT #
        # ------ #
        subparser.add_argument('--type_draw',
                               type=str,
                               choices=parameters_choices['type_draw'],
                               default=parameters_default['type_draw'],
                               help=parameters_help['type_draw'])

        subparser.add_argument('--do_output_gravity',
                               action='store_true',
                               default=parameters_default['do_output_gravity'],
                               help=parameters_help['do_output_gravity'])

        subparser.add_argument('--num_images',
                               type=int,
                               default=parameters_default['num_images'],
                               help=parameters_help['num_images'])

        subparser.add_argument('--idx',
                               type=int,
                               default=parameters_default['idx'],
                               help=parameters_help['idx'])

        # ---------- #
        # OUTPUT FPS #
        # ---------- #
        subparser.add_argument('--FPS',
                               type=int,
                               default=parameters_default['FPS'],
                               help=parameters_help['FPS'])

        # --------------- #
        # POST PROCESSING #
        # --------------- #
        subparser.add_argument('--do_NMS',
                               action='store_true',
                               default=parameters_default['do_NMS'],
                               help=parameters_help['do_NMS'])

        subparser.add_argument('--NMS_box_radius',
                               type=int,
                               default=parameters_default['NMS_box_radius'],
                               help=parameters_help['NMS_box_radius'])

        # ------ #
        # ROCalc #
        # ------ #
        subparser.add_argument('--type_detections',
                               type=str,
                               default=parameters_default['type_detections'],
                               help=parameters_help['type_detections'])

        # ---------- #
        # PLOT CHECK #
        # ---------- #
        subparser.add_argument('--plot_check_list',
                               type=str,
                               default=parameters_default['plot_check_list'],
                               help=parameters_help['plot_check_list'])

        subparser.add_argument('--type_plot_check',
                               type=str,
                               default=parameters_default['type_plot_check'],
                               help=parameters_help['type_plot_check'])

        subparser.add_argument('--do_plots_train',
                               action='store_true',
                               default=parameters_default['do_plots_train'],
                               help=parameters_help['do_plots_train'])

        subparser.add_argument('--do_plots_validation',
                               action='store_true',
                               default=parameters_default['do_plots_validation'],
                               help=parameters_help['do_plots_validation'])

        subparser.add_argument('--do_plots_test',
                               action='store_true',
                               default=parameters_default['do_plots_test'],
                               help=parameters_help['do_plots_test'])

        subparser.add_argument('--do_plots_test_NMS',
                               action='store_true',
                               default=parameters_default['do_plots_test_NMS'],
                               help=parameters_help['do_plots_test_NMS'])

        subparser.add_argument('--do_plots_test_all',
                               action='store_true',
                               default=parameters_default['do_plots_test_all'],
                               help=parameters_help['do_plots_test_all'])

        subparser.add_argument('--do_metrics',
                               action='store_true',
                               default=parameters_default['do_metrics'],
                               help=parameters_help['do_metrics'])

        subparser.add_argument('--do_plots',
                               action='store_true',
                               default=parameters_default['do_plots'],
                               help=parameters_help['do_plots'])

        # ----- #
        # DEBUG #
        # ----- #
        subparser.add_argument('--debug_execution',
                               action='store_true',
                               default=parameters_default['debug_execution'],
                               help=parameters_help['debug_execution'])

        subparser.add_argument('--debug_initialization',
                               action='store_true',
                               default=parameters_default['debug_initialization'],
                               help=parameters_help['debug_initialization'])

        subparser.add_argument('--debug_transforms',
                               action='store_true',
                               default=parameters_default['debug_transforms'],
                               help=parameters_help['debug_transforms'])

        subparser.add_argument('--debug_transforms_augmentation',
                               action='store_true',
                               default=parameters_default['debug_transforms_augmentation'],
                               help=parameters_help['debug_transforms_augmentation'])

        subparser.add_argument('--debug_anchors',
                               action='store_true',
                               default=parameters_default['debug_anchors'],
                               help=parameters_help['debug_anchors'])

        subparser.add_argument('--debug_hooking',
                               action='store_true',
                               default=parameters_default['debug_hooking'],
                               help=parameters_help['debug_hooking'])

        subparser.add_argument('--debug_network',
                               action='store_true',
                               default=parameters_default['debug_network'],
                               help=parameters_help['debug_network'])

        subparser.add_argument('--debug_test',
                               action='store_true',
                               default=parameters_default['debug_test'],
                               help=parameters_help['debug_test'])

        subparser.add_argument('--debug_validation',
                               action='store_true',
                               default=parameters_default['debug_validation'],
                               help=parameters_help['debug_validation'])

        subparser.add_argument('--debug_FROC',
                               action='store_true',
                               default=parameters_default['debug_FROC'],
                               help=parameters_help['debug_FROC'])

    # parser arguments
    parser = parser.parse_args()

    return parser
