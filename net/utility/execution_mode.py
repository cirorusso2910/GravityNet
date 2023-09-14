import sys

from net.utility.msg.msg_error import msg_error
from net.utility.my_creator import my_creator


def execution_mode(mode: str,
                   option: str):
    """
    Show execution mode status

    :param mode: execution mode
    :param option: execution option
    :return: show execution status
    """

    # who is my creator
    my_creator(mode=mode)

    if option == 'start':
        print("Execution Mode: {}".format(mode.upper()))

    elif option == 'complete':
        print("\nExecution Mode: {} Complete".format(mode.upper()))

    else:
        str_err = msg_error(file=__file__,
                            variable=option,
                            type_variable='option',
                            choices='[start, complete]')
        sys.exit(str_err)
