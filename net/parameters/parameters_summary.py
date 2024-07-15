import argparse

from typing import Tuple

import numpy as np

from net.evaluation.utility.distance_eval_rescale import distance_eval_rescale
from net.metrics.utility.my_notation import scientific_notation


def parameters_summary(parser: argparse.Namespace,
                       dataset_name: str,
                       num_images: dict,
                       num_images_normals: dict,
                       num_annotations: dict,
                       small_lesion: str,
                       image_shape: np.array,
                       feature_map_shape: Tuple[int, int],
                       num_gravity_points_feature_map: int,
                       num_gravity_points: int):
    """
    Show parameters summary

    :param parser: parser of parameters-parsing
    :param dataset_name: dataset name
    :param num_images: num images dictionary
    :param num_images_normals: num images normals dictionary
    :param num_annotations: num annotations dictionary
    :param small_lesion: small lesion type
    :param image_shape: image shape
    :param feature_map_shape: feature map shape
    :param num_gravity_points_feature_map: num gravity points per feature map
    :param num_gravity_points: num gravity points
    """

    print("\n-------------------"
          "\nPARAMETERS SUMMARY:"
          "\n-------------------")

    # ------- #
    # DATASET #
    # ------- #
    print("\nDATASET:"
          "\nDataset name: {}".format(dataset_name),
          "\nData Split: {}".format(parser.split),
          "\nTraining data: {} with {} {}".format(num_images['train'], num_annotations['train'], small_lesion),
          "\nValidation data: {} with {} {}".format(num_images['validation'], num_annotations['validation'], small_lesion),
          "\nTest data: {} with {} {}".format(num_images['test'], num_annotations['test'], small_lesion))

    print("\nIMAGE NORMALS:"
          "\nTraining image normals: {}".format(num_images_normals['train']),
          "\nValidation image normals: {}".format(num_images_normals['validation']),
          "\nTest image normals: {}".format(num_images_normals['test']))

    # ------------------ #
    # DATASET TRANSFORMS #
    # ------------------ #
    print("\nDATASET TRANSFORMS:"
          "\nRescale factor: {} ({} %)".format(parser.rescale, int(parser.rescale * 100)),
          "\nMax padding: {}".format(parser.max_padding))

    # -------------------- #
    # DATASET AUGMENTATION #
    # -------------------- #
    if parser.do_dataset_augmentation:
        print("\nDATASET AUGMENTED:"
              "\nTraining data augmented: {} with {} {}".format(num_images['train'] * 4, num_annotations['train'] * 4, small_lesion))

        print("\nDATASET TRANSFORMS AUGMENTATION:"
              "\nHorizontal Flip"
              "\nVertical Flip"
              "\nHorizontal and Vertical Flip")

    # --------------------- #
    # DATASET NORMALIZATION #
    # --------------------- #
    print("\nDATASET NORMALIZATION:"
          "\nNormalization: {}".format(parser.norm))

    # ---------- #
    # DATALOADER #
    # ---------- #
    print("\nDATALOADER:"
          "\nBatch size train: {}".format(parser.batch_size_train),
          "\nBatch size validation: {}".format(parser.batch_size_val),
          "\nBatch size test: {}".format(parser.batch_size_test))

    # -------------- #
    # GRAVITY POINTS #
    # -------------- #
    print("\nGRAVITY POINTS:"
          "\nConfiguration: {}".format(parser.config),
          "\nImage shape (H x W): {} x {}".format(image_shape[0], image_shape[1]),
          "\nFeature map shape (H x W): {} x {}".format(feature_map_shape[0], feature_map_shape[1]),
          "\nGravity Points for feature map: {}".format(num_gravity_points_feature_map),
          "\nGravity Points for image: {}".format(num_gravity_points))

    # ------------- #
    # NETWORK MODEL #
    # ------------- #
    print("\nNETWORK MODEL:"
          "\nBackbone: {}".format(parser.backbone),
          "\nPreTrained: {}".format(parser.pretrained))

    # ---------------- #
    # HYPER PARAMETERS #
    # ---------------- #
    print("\nHYPER PARAMETERS:"
          "\nEpochs: {}".format(parser.epochs))

    print("Optimizer: {}".format(parser.optimizer),
          "\nScheduler: {}".format(parser.scheduler),
          "\nClip Gradient: {}".format(parser.clip_gradient),
          "\nLearning Rate: {}".format(scientific_notation(parser.learning_rate)))

    if parser.optimizer == 'SGD':
        print("Momentum: {}".format(parser.lr_momentum))

    if parser.scheduler == 'ReduceLROnPlateau':
        print("Patience: {}".format(parser.lr_patience))

    elif parser.scheduler == 'StepLR':
        print("Step Size: {}".format(parser.lr_step_size))

    elif parser.scheduler == 'CosineAnnealing':
        print("T0: {}".format(parser.lr_T0))

    if parser.clip_gradient:
        print("Max Norm: {}".format(parser.max_norm))

    # --------- #
    # CRITERION #
    # --------- #
    print("\nCRITERION:"
          "\nAlpha: {}".format(parser.alpha),
          "\nGamma: {}".format(parser.gamma),
          "\nHook distance: {} pixel".format(parser.hook))

    # ---------- #
    # EVALUATION #
    # ---------- #
    # distance
    print("\nEVALUATION:"
          "\nEval: {}".format(parser.eval),
          "\nEval with rescale of {} ({} %): distance{}".format(parser.rescale, int(parser.rescale * 100), distance_eval_rescale(eval=parser.eval, rescale=parser.rescale)),
          "\nFP Images: {}".format(parser.FP_images))

    # ---------- #
    # LOAD MODEL #
    # ---------- #
    if parser.mode in ['test', 'test_NMS']:
        print("\nLOAD MODEL:")
        if parser.load_best_sensitivity_10_FPS_model:
            print("Load best sensitivity 10 FPS model")
        if parser.load_best_AUFROC_0_10_model:
            print("Load best AUFROC [0, 10] model")
        if parser.load_best_AUPR_model:
            print("Load best AUPR model")

    # ------ #
    # OUTPUT #
    # ------ #
    print("\nOUTPUT:"
          "\nType draw: {}".format(parser.type_draw),
          "\nOutput gravity: {}".format(parser.do_output_gravity))
    if parser.mode in ['train']:
        print("Num images (in validation): 1")
    if parser.mode in ['test']:
        print("Num images (in test): {}".format(parser.num_images))

    # ---------------------------- #
    # NON-MAXIMA-SUPPRESSION (NMS) #
    # ---------------------------- #
    if parser.do_NMS:
        print("\nNON-MAXIMA-SUPPRESSION (NMS):"
              "\nNMS box radius: {}".format(parser.NMS_box_radius))
