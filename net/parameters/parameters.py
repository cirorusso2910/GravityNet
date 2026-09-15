import argparse
import sys

import yaml

from net.parameters.parameters_choices import parameters_choices
from net.parameters.parameters_default import parameters_default
from net.parameters.parameters_help import parameters_help


def resolve_effective_defaults() -> dict:
    """
    Pre-parse '--cfg' (a YAML file of default overrides) so its values can be used as the
    argparse 'default=' below: CLI flags parsed afterwards still win over them, and YAML values
    still win over parameters_default. Unknown keys in the YAML file are a hard error (typo guard).

    :return: parameters_default merged with the YAML overrides (or a plain copy if no --cfg given)
    """

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--cfg', type=str, default=None)
    cfg_path = pre_parser.parse_known_args()[0].cfg

    effective_defaults = dict(parameters_default)

    if cfg_path:
        with open(cfg_path) as file:
            overrides = yaml.safe_load(file) or {}

        unknown_keys = sorted(set(overrides) - set(parameters_default))
        if unknown_keys:
            sys.exit(f"ERROR: unknown key(s) in --cfg '{cfg_path}': {unknown_keys}")

        effective_defaults.update(overrides)

    return effective_defaults


def parameters_parsing() -> argparse.Namespace:
    """
    Definition of parameters-parsing for each execution mode

    :return: parser of parameters parsing
    """

    # CLI > YAML (--cfg) > parameters_default
    parameters_default = resolve_effective_defaults()

    # parser
    parser = argparse.ArgumentParser(description='Argument Parser')

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    parser_mode = parser.add_subparsers(title=parameters_help['mode'], dest='mode', metavar='mode')

    # execution mode
    parser_train = parser_mode.add_parser('train', help=parameters_help['train'])
    parser_test = parser_mode.add_parser('test', help=parameters_help['test'])
    parser_train_test = parser_mode.add_parser('train_test', help=parameters_help['train_test'])

    parser_explainability = parser_mode.add_parser('explainability', help=parameters_help['explainability'])

    # who is my creator
    parser_who_is_my_creator = parser_mode.add_parser('who_is_my_creator', help=parameters_help['who_is_my_creator'])

    # execution mode script
    parser_script_anchors = parser_mode.add_parser('script_anchors', help=parameters_help['script_anchors'])
    parser_script_dataset = parser_mode.add_parser('script_dataset', help=parameters_help['script_dataset'])

    # execution mode list
    execution_mode = [parser_train,
                      parser_test,
                      parser_train_test,
                      parser_explainability,
                      parser_script_anchors,
                      parser_script_dataset,
                      parser_who_is_my_creator]

    # for each subparser 'mode'
    for subparser in execution_mode:

        # ------ #
        # CONFIG #
        # ------ #
        subparser.add_argument('--cfg',
                               type=str,
                               default=None,
                               help="path to a YAML file overriding parameter defaults (CLI flags still take precedence)")

        # -------------- #
        # INITIALIZATION #
        # -------------- #
        subparser.add_argument('--dataset_path',
                               type=str,
                               default=parameters_default['dataset_path'],
                               help=parameters_help['dataset_path'])

        subparser.add_argument('--experiments_path',
                               type=str,
                               default=parameters_default['experiments_path'],
                               help=parameters_help['experiments_path'])

        # ------------ #
        # LOAD DATASET #
        # ------------ #
        subparser.add_argument('--dataset',
                               type=str,
                               default=parameters_default['dataset'],
                               help=parameters_help['dataset'])

        subparser.add_argument('--small_lesion',
                               type=str,
                               default=parameters_default['small_lesion'],
                               help=parameters_help['small_lesion'])

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

        # --------------- #
        # UTILITY DATASET #
        # --------------- #
        subparser.add_argument('--images_extension',
                               type=str,
                               default=parameters_default['images_extension'],
                               help=parameters_help['images_extension'])

        subparser.add_argument('--images_masks_extension',
                               type=str,
                               default=parameters_default['images_masks_extension'],
                               help=parameters_help['images_masks_extension'])

        subparser.add_argument('--annotations_extension',
                               type=str,
                               default=parameters_default['annotations_extension'],
                               help=parameters_help['annotations_extension'])

        # ------------- #
        # EXPERIMENT ID #
        # ------------- #
        subparser.add_argument('--typeID',
                               type=str,
                               default=parameters_default['typeID'],
                               help=parameters_help['typeID'])

        subparser.add_argument('--sep',
                               type=str,
                               default=parameters_default['sep'],
                               help=parameters_help['sep'])

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
        subparser.add_argument('--rescale',
                               type=float,
                               default=parameters_default['rescale'],
                               help=parameters_help['rescale'])

        subparser.add_argument('--num_channels',
                               type=int,
                               default=parameters_default['num_channels'],
                               choices=parameters_choices['num_channels'],
                               help=parameters_help['num_channels'])

        subparser.add_argument('--max_padding',
                               type=int,
                               default=parameters_default['max_padding'],
                               help=parameters_help['max_padding'])

        # -------------------- #
        # DATASET AUGMENTATION #
        # -------------------- #
        subparser.add_argument('--do_dataset_augmentation',
                               action=argparse.BooleanOptionalAction,
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
                               action=argparse.BooleanOptionalAction,
                               default=parameters_default['pretrained'],
                               help=parameters_help['pretrained'])

        # -------------- #
        # GRAVITY POINTS #
        # -------------- #
        subparser.add_argument('--config',
                               type=str,
                               default=parameters_default['config'],
                               help=parameters_help['config'])

        subparser.add_argument('--save_config',
                               action=argparse.BooleanOptionalAction,
                               default=parameters_default['save_config'],
                               help=parameters_help['save_config'])

        # ---------------- #
        # HYPER-PARAMETERS #
        # ---------------- #
        subparser.add_argument('--epochs', '--ep',
                               type=int,
                               default=parameters_default['epochs'],
                               help=parameters_help['epochs'])

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
                               action=argparse.BooleanOptionalAction,
                               default=parameters_default['clip_gradient'],
                               help=parameters_help['clip_gradient'])

        subparser.add_argument('--learning_rate', '--lr',
                               type=float,
                               default=parameters_default['learning_rate'],
                               help=parameters_help['learning_rate'])

        subparser.add_argument('--lr_momentum',
                               type=float,
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
                               type=float,
                               default=parameters_default['lr_gamma'],
                               help=parameters_help['lr_gamma'])

        subparser.add_argument('--lr_T0',
                               type=int,
                               default=parameters_default['lr_T0'],
                               help=parameters_help['lr_T0'])

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
                               type=float,
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

        subparser.add_argument('--score_threshold',
                               type=float,
                               default=parameters_default['score_threshold'],
                               help=parameters_help['score_threshold'])

        # ---------- #
        # LOAD MODEL #
        # ---------- #
        subparser.add_argument('--load_best_sensitivity_10_FPS_model',
                               action=argparse.BooleanOptionalAction,
                               default=parameters_default['load_best_sensitivity_10_FPS_model'],
                               help=parameters_help['load_best_sensitivity_10_FPS_model'])

        subparser.add_argument('--load_best_AUFROC_0_10_model',
                               action=argparse.BooleanOptionalAction,
                               default=parameters_default['load_best_AUFROC_0_10_model'],
                               help=parameters_help['load_best_AUFROC_0_10_model'])

        subparser.add_argument('--load_best_AUPR_model',
                               action=argparse.BooleanOptionalAction,
                               default=parameters_default['load_best_AUPR_model'],
                               help=parameters_help['load_best_AUPR_model'])

        # ------ #
        # OUTPUT #
        # ------ #
        subparser.add_argument('--type_draw',
                               type=str,
                               choices=parameters_choices['type_draw'],
                               default=parameters_default['type_draw'],
                               help=parameters_help['type_draw'])

        subparser.add_argument('--box_draw_radius',
                               type=int,
                               default=parameters_default['box_draw_radius'],
                               help=parameters_help['box_draw_radius'])

        subparser.add_argument('--do_output_gravity',
                               action=argparse.BooleanOptionalAction,
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

        # --------------- #
        # POST PROCESSING #
        # --------------- #
        subparser.add_argument('--do_NMS',
                               action=argparse.BooleanOptionalAction,
                               default=parameters_default['do_NMS'],
                               help=parameters_help['do_NMS'])

        subparser.add_argument('--NMS_box_radius',
                               type=int,
                               default=parameters_default['NMS_box_radius'],
                               help=parameters_help['NMS_box_radius'])

    # parser arguments
    parser = parser.parse_args()

    return parser
