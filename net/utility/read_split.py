from pandas import read_csv


def read_split(path_split: str):
    """
    Read data split

    :param path_split: path split
    :return: split dictionary
    """

    # read csv
    images_split = read_csv(filepath_or_buffer=path_split, usecols=["INDEX", "FILENAME", "SPLIT"]).values

    index = images_split[:, 0]
    filename = images_split[:, 1]
    split = images_split[:, 2]

    split_dict = {
        'index': index.tolist(),
        'filename': filename.tolist(),
        'split': split.tolist()
    }

    return split_dict
