import argparse
import sys
import torch

from net.utility.msg.msg_load_model_complete import msg_load_best_model_complete


def load_best_model(net: torch.nn.Module,
                    metrics_type: str,
                    path: str):
    """
    Load best-model

    :param net: net
    :param metrics_type: metrics type
    :param path: path
    """

    # load model
    load_model = torch.load(path)

    # load state dict
    net.load_state_dict(load_model['net_state_dict'])

    # msg load best model complete
    msg_load_best_model_complete(metrics_type=metrics_type,
                                 load_model=load_model)


def check_load_model(parser: argparse.Namespace):
    """
    Check load model

    :param parser: parser of parameters-parsing
    """

    check = True

    if parser.load_best_sensitivity_model and parser.load_best_AUFROC_model:
        check = False

    if parser.load_best_sensitivity_model and not parser.load_best_AUFROC_model:
        check = True

    if parser.load_best_AUFROC_model and not parser.load_best_sensitivity_model:
        check = True

    if not parser.load_best_sensitivity_model and not parser.load_best_AUFROC_model:
        check = False

    if not check:
        print(parser.load_best_sensitivity_model)
        print(parser.load_best_AUFROC_model)
        sys.exit("ERROR")
