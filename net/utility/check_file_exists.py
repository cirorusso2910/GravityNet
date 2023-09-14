import os


def check_file_exists(path: str,
                      filename: str):
    """
    Check if file exists

    :param path: path
    :param filename: filename
    """

    if not os.path.exists(path):
        print("WARNING: {} not exist".format(filename))
