import argparse

from net.device.get_GPU_name import get_GPU_name
from net.metrics.utility.my_notation import scientific_notation


def parameters_ID(parser: argparse.Namespace) -> dict:
    """
    Get parameters ID

    :param parser: parser of parameters-parsing
    :return: parameters ID dictionary
    """

    # ------------- #
    # PARAMETERS ID #
    # ------------- #
    dataset_ID = "dataset={}".format(parser.dataset)
    dataset_augmented_ID = "dataset={}-augmented".format(parser.dataset)
    norm_ID = "norm={}".format(parser.norm)
    split_ID = "split={}".format(parser.split)
    rescale_ID = "rescale={}".format(parser.rescale)
    ep_ID = "ep={}".format(parser.epochs)
    lr_ID = "lr={}".format(scientific_notation(parser.learning_rate))
    bs_ID = "bs={}".format(parser.batch_size_train)
    backbone_ID = "backbone={}".format(parser.backbone)
    pretrained_ID = "pretrained={}".format(parser.pretrained)
    config_ID = "config={}".format(parser.config)
    hook_ID = "hook={}".format(parser.hook)
    eval_ID = "eval={}".format(parser.eval)
    if 'script' in parser.mode:
        GPU_ID = "GPU={}".format(parser.GPU)
    else:
        GPU_ID = "GPU={}".format(get_GPU_name())

    parameters_ID_dict = {
        'dataset': dataset_ID,
        'dataset_augmented': dataset_augmented_ID,
        'norm': norm_ID,
        'split': split_ID,
        'rescale': rescale_ID,
        'ep': ep_ID,
        'lr': lr_ID,
        'bs': bs_ID,
        'backbone': backbone_ID,
        'pretrained': pretrained_ID,
        'config': config_ID,
        'hook': hook_ID,
        'eval': eval_ID,
        'GPU': GPU_ID,
    }

    return parameters_ID_dict
