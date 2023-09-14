import argparse
import sys

from net.initialization.utility.parameters_ID import parameters_ID
from net.utility.msg.msg_error import msg_error


def experimentID_complete(typeID: str,
                          parser: argparse.Namespace) -> str:
    """
    Concatenate experiment-ID complete

    :param typeID: type experiment-ID
    :param parser: parser of parameters-parsing
    :return: experiment-ID complete
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
        experiment_complete_ID = parameters_ID_dict['dataset'] + "|" + "split=default" + "|" + parameters_ID_dict['rescale'] + "|" + parameters_ID_dict['channel'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['config'] + "|" + parameters_ID_dict['hook'] + "|" + parameters_ID_dict['eval'] + "|" + parameters_ID_dict['GPU']

    # ---------- #
    # NO-HEALTHY #
    # ---------- #
    elif typeID == 'no-healthy':
        experiment_complete_ID = parameters_ID_dict['dataset'] + "|" + "split=no-healthy" + "|" + parameters_ID_dict['rescale'] + "|" + parameters_ID_dict['channel'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['config'] + "|" + parameters_ID_dict['hook'] + "|" + parameters_ID_dict['eval'] + "|" + parameters_ID_dict['GPU']

    else:
        str_err = msg_error(file=__file__,
                            variable=parser.typeID,
                            type_variable='type experiment complete ID',
                            choices='[default, no-healthy]')
        sys.exit(str_err)

    return experiment_complete_ID
