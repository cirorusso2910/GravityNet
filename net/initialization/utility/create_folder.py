import os


def create_folder(path: str):
    """
    Create folder

    :param path: path
    """

    # create experiment result folder
    if not os.path.exists(path):
        os.mkdir(path)