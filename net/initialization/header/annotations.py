import sys

from net.utility.msg.msg_error import msg_error


def annotation_header(annotation_type: str) -> list:
    """
    Annotation header

    :param dataset: dataset name
    :param annotation_type: annotation type
    :return: header
    """


    # default
    if annotation_type == 'default':
        header = ["X", "Y", "RADIUS"]

    else:
        str_err = msg_error(file=__file__,
                        variable=annotation_type,
                        type_variable='$DATASET$ annotation-type',
                        choices='[default')
        sys.exit(str_err)

    return header
