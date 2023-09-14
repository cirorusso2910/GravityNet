import sys

from net.utility.msg.msg_error import msg_error


def annotation_header(dataset: str,
                      annotation_type: str) -> list:
    """
    Annotation header

    :param dataset: dataset name
    :param annotation_type: annotation type
    :return: header
    """

    # INbreast
    if dataset == 'INbreast':

        # default
        if annotation_type == 'default':
            header = ["ROI", "X", "Y", "RADIUS"]

        # w48m14
        elif annotation_type == 'w48m14':
            header = ["ROI", "X", "Y"]

        else:
            str_err = msg_error(file=__file__,
                                variable=annotation_type,
                                type_variable='INbreast annotation-type',
                                choices='[default, w48m14')
            sys.exit(str_err)

    # E-ophtha-MA
    elif dataset == 'E-ophtha-MA':

        # default
        if annotation_type == 'default':
            header = ["X", "Y", "RADIUS X", "RADIUS Y"]

        else:
            str_err = msg_error(file=__file__,
                                variable=annotation_type,
                                type_variable="annotation type",
                                choices="[default]")
            sys.exit(str_err)

    else:
        str_err = msg_error(file=__file__,
                            variable=dataset,
                            type_variable="dataset",
                            choices="[INbreast, E-ophtha-MA]")
        sys.exit(str_err)

    return header
