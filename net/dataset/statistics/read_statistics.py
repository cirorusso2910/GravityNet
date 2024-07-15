from pandas import read_csv

from net.initialization.header.statistics import statistics_header


def read_statistics(statistics_path: str,
                    small_lesion: str) -> dict:
    """
    Read statistics

    :param statistics_path: statistics path
    :param small_lesion: small lesion
    :return: statistics dictionary
    """

    # get statistics header
    header = statistics_header(statistics_type='statistics',
                               small_lesion_type=small_lesion)

    # read statistics
    statistics = read_csv(filepath_or_buffer=statistics_path, usecols=header)

    statistics_dict = {
        'images': {
            'train': statistics['IMAGES'][0],
            'validation': statistics['IMAGES'][1],
            'test': statistics['IMAGES'][2],
        },
        'normals': {
            'train': statistics['NORMALS'][0],
            'validation': statistics['NORMALS'][1],
            'test': statistics['NORMALS'][2],
        },
        'annotations': {
            'train': statistics['{}'.format(small_lesion.upper())][0],
            'validation': statistics['{}'.format(small_lesion.upper())][1],
            'test': statistics['{}'.format(small_lesion.upper())][2],
        },
    }

    return statistics_dict
