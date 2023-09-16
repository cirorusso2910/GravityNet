import sys

from net.utility.msg.msg_error import msg_error


def default_folders_dict(where: str) -> dict:
    """
    Default folders dictionary

    :param where: where
    :return: default folders dictionary
    """

    if where == '$WHERE$':
        default_folder = {
            'datasets': "$PATH$/datasets",

            'doc': "$PATH$/GravityNet/doc",

            'experiments': "$PATH$/experiments",
            'experiments_complete': "$PATH$/experiments-complete",

            'plot_check': "$PATH$/plot-check",
            'plot_check_complete': "$PATH$/plot-check-complete",
        }

    else:
        str_err = msg_error(file=__file__,
                            variable=where,
                            type_variable='where',
                            choices='[$WHERE$]')
        sys.exit(str_err)

    return default_folder

