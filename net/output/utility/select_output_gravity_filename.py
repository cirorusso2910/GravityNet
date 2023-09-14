import sys

from net.utility.msg.msg_error import msg_error


def select_output_gravity_filename(dataset: str,
                                   split: str) -> str:
    """
    Select filename to save output gravity

    :param dataset: dataset name
    :param split: split
    :return: filename for output gravity in validation
    """

    # INbreast
    if dataset == 'INbreast':
        # image filename for output gravity
        if split == '1-fold':
            filename_output_gravity = '50997461_97ec8cadfca70d32_MG_R_ML_ANON'
        elif split == '2-fold':
            filename_output_gravity = '22678495_60995d51033e24b8_MG_R_ML_ANON'
        else:
            filename_output_gravity = ""
            print("\nNO FILENAME OUTPUT GRAVITY!")

    # E-ophtha-MA
    elif dataset == 'E-ophtha-MA':
        # image filename for output gravity
        if split == '1-fold':
            filename_output_gravity = 'DS000HXJ'
        elif split == '2-fold':
            filename_output_gravity = 'C0007157'
        elif split == 'debug':
            filename_output_gravity = 'DS000HXJ'
        else:
            filename_output_gravity = ""
            print("\nWARNING: NO FILENAME OUTPUT GRAVITY!")

    else:
        str_err = msg_error(file=__file__,
                            variable=dataset,
                            type_variable="dataset",
                            choices="[INbreast, E-ophtha-MA]")
        sys.exit(str_err)

    return filename_output_gravity
