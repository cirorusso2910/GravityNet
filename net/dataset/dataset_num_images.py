from net.dataset.statistics.read_statistics import read_statistics


def dataset_num_images(statistics_path: str,
                       small_lesion: str) -> dict:
    """
    Compute dataset num images

    :param statistics_path: statistics path
    :param small_lesion: small lesion
    :return: dataset num images dictionary
    """

    # read statistics
    statistics = read_statistics(statistics_path=statistics_path,
                                 small_lesion=small_lesion)

    num_images = {
        'train': statistics['images']['train'],
        'validation': statistics['images']['validation'],
        'test': statistics['images']['test']
    }

    return num_images
