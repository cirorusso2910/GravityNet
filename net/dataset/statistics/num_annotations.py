from torch.utils.data import Dataset


def get_num_annotations(dataset: Dataset) -> int:
    """
    Get num annotations

    :param dataset: dataset
    :return: num annotations
    """

    # init
    tot_annotation = 0

    # dataset size
    dataset_size = dataset.__len__()

    for i in range(dataset_size):
        annotation = dataset[i]['annotation']
        annotation = annotation[annotation[:, 0] != -1]  # delete padding -1

        num_annotation = len(annotation)

        tot_annotation += num_annotation

    return tot_annotation
