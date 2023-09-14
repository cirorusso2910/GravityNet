from torch.utils.data import Dataset


def get_num_normal_images(dataset: Dataset,
                          annotation_type_dict: str) -> int:
    """
    Get num normal images

    :param dataset: dataset
    :param annotation_type_dict: annotation type
    :return: num normal images
    """

    # init
    tot_normal_images = 0

    # dataset size
    dataset_size = dataset.__len__()

    for i in range(dataset_size):
        annotation = dataset[i][annotation_type_dict]
        num_annotation = len(annotation)

        if num_annotation == 0:
            tot_normal_images += 1

    return tot_normal_images
