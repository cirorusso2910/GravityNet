from typing import List


def read_file(file_path: str) -> List:
    """
    Read file

    :param file_path: path of file
    :return: file lines
    """

    file_lines = []
    with open(file_path, 'r') as fid:
        for line in fid:
            file_lines.append(line.rstrip())

    return file_lines
