from typing import Any


def msg_error(file: str,
              variable: Any,
              type_variable: str,
              choices: str) -> str:
    """
    Concatenate message string error:

    ERROR in 'file.py'
    'variable' wrong in 'type_variable'
    '[choiches of variable]'

    :param file: file where error occur
    :param variable: variable
    :param type_variable: type of variable
    :param choices: choiches of variable
    :return: string error
    """

    str_err = "\nERROR in {}" \
              "\n{} wrong {}" \
              "\n{}".format(file, variable, type_variable, choices)

    return str_err
