from net.dataset.statistics.read_statistics import read_statistics


def dataset_num_normal_images(statistics_path: str,
                              small_lesion: str) -> dict:
    """
    Compute dataset num normal images

    :param statistics_path: statistics path
    :param small_lesion: small lesion
    :return: dataset num images normals dictionary
    """

    # read statistics
    statistics = read_statistics(statistics_path=statistics_path,
                                 small_lesion=small_lesion)

    num_normal_images = {
        'train': statistics['normals']['train'],
        'validation': statistics['normals']['validation'],
        'test': statistics['normals']['test'],
    }

    return num_normal_images
