import sys

from net.utility.msg.msg_error import msg_error


def default_folders_dict(where: str) -> dict:
    """
    Default folders dictionary

    :param where: where [home, data]
    :return: default folders dictionary
    """

    if where == 'home':
        default_folder = {
            'datasets': "/data/russo/datasets",

            'doc': "/home/russo/GravityNet/doc",

            'experiments': "/home/russo/GravityNet/experiments",
            'experiments_complete': "/home/russo/GravityNet/experiments-complete",

            'plot_check': "/home/russo/GravityNet/plot-check",
            'plot_check_complete': "/home/russo/GravityNet/plot-check-complete",
        }

    elif where == 'data':
        default_folder = {
            'datasets': "/data/russo/datasets",

            'doc': "/home/russo/GravityNet/doc",

            'experiments': "/data/russo/GravityNet/experiments",
            'experiments_complete': "/data/russo/GravityNet/experiments-complete",

            'plot_check': "/home/russo/GravityNet/plot-check",
            'plot_check_complete': "/home/russo/GravityNet/plot-check-complete",
        }

    else:
        str_err = msg_error(file=__file__,
                            variable=where,
                            type_variable='where',
                            choices='[home, data]')
        sys.exit(str_err)

    return default_folder

