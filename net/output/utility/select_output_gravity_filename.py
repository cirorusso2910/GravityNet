def select_output_gravity_filename(split: str) -> str:
    """
    Select filename to save output gravity

    :param dataset: dataset name
    :param split: split
    :return: filename for output gravity in validation
    """

    # image filename for output gravity
    if split == '1-fold':
        filename_output_gravity = '$FILENAME$'
    elif split == '2-fold':
        filename_output_gravity = '$FILENAME$'
    else:
        filename_output_gravity = ""
        print("\nNO FILENAME OUTPUT GRAVITY!")

    return filename_output_gravity
