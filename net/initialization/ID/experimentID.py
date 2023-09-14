import argparse
import sys

from typing import Tuple

from net.initialization.utility.parameters_ID import parameters_ID
from net.utility.msg.msg_error import msg_error


def experimentID(typeID: str,
                 parser: argparse.Namespace) -> Tuple[str, str]:
    """
    Concatenate experiment-ID according to type

    :param typeID: type experiment-ID
    :param parser: parser of parameters-parsing
    :return: experiment-ID and experiment-ID for resume
    """

    # ------------- #
    # PARAMETERS ID #
    # ------------- #
    # parameters ID dictionary
    parameters_ID_dict = parameters_ID(parser=parser)

    # ------- #
    # DEFAULT #
    # ------- #
    if typeID == 'default':
        experiment_ID = parameters_ID_dict['dataset'] + "|" + parameters_ID_dict['split'] + "|" + parameters_ID_dict['rescale'] + "|" + parameters_ID_dict['channel'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['config'] + "|" + parameters_ID_dict['hook'] + "|" + parameters_ID_dict['eval'] + "|" + parameters_ID_dict['GPU']
        experiment_resume_ID = parameters_ID_dict['dataset'] + "|" + parameters_ID_dict['split'] + "|" + parameters_ID_dict['rescale'] + "|" + parameters_ID_dict['channel'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep_to_resume'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['config'] + "|" + parameters_ID_dict['hook'] + "|" + parameters_ID_dict['eval'] + "|" + parameters_ID_dict['GPU']

        if parser.do_dataset_augmentation:
            experiment_ID = parameters_ID_dict['dataset_augmented'] + "|" + parameters_ID_dict['split'] + "|" + parameters_ID_dict['rescale'] + "|" + parameters_ID_dict['channel'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['config'] + "|" + parameters_ID_dict['hook'] + "|" + parameters_ID_dict['eval'] + "|" + parameters_ID_dict['GPU']
            experiment_resume_ID = parameters_ID_dict['dataset_augmented'] + "|" + parameters_ID_dict['split'] + "|" + parameters_ID_dict['rescale'] + "|" + parameters_ID_dict['channel'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep_to_resume'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['config'] + "|" + parameters_ID_dict['hook'] + "|" + parameters_ID_dict['eval'] + "|" + parameters_ID_dict['GPU']

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.typeID,
                            type_variable='type experiment ID',
                            choices='[default]')
        sys.exit(str_err)

    return experiment_ID, experiment_resume_ID
