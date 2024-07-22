import argparse
import sys

from net.initialization.utility.parameters_ID import parameters_ID
from net.utility.msg.msg_error import msg_error


def experimentID(typeID: str,
                 sep: str,
                 parser: argparse.Namespace) -> str:
    """
    Concatenate experiment-ID according to type

    :param typeID: type experiment-ID
    :param sep: separator
    :param parser: parser of parameters-parsing
    :return: experiment-ID
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
        experiment_ID = parameters_ID_dict['dataset'] + sep + parameters_ID_dict['split'] + sep + parameters_ID_dict['rescale'] + sep + parameters_ID_dict['norm'] + sep + parameters_ID_dict['ep'] + sep + parameters_ID_dict['lr'] + sep + parameters_ID_dict['bs'] + sep + parameters_ID_dict['backbone'] + sep + parameters_ID_dict['config'] + sep + parameters_ID_dict['hook'] + sep + parameters_ID_dict['eval'] + sep + parameters_ID_dict['GPU']

        if parser.do_dataset_augmentation:
            experiment_ID = parameters_ID_dict['dataset_augmented'] + sep + parameters_ID_dict['split'] + sep + parameters_ID_dict['rescale'] + sep + parameters_ID_dict['norm'] + sep + parameters_ID_dict['ep'] + sep + parameters_ID_dict['lr'] + sep + parameters_ID_dict['bs'] + sep + parameters_ID_dict['backbone'] + sep + parameters_ID_dict['config'] + sep + parameters_ID_dict['hook'] + sep + parameters_ID_dict['eval'] + sep + parameters_ID_dict['GPU']

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.typeID,
                            type_variable='type experiment ID',
                            choices='[default]')
        sys.exit(str_err)

    return experiment_ID
