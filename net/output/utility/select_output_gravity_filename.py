from torch.utils.data import Dataset


def select_output_gravity_filename(dataset: Dataset,
                                   idx: int) -> str:
    """
    Select filename to save output gravity

    :param dataset: dataset name
    :param split: split
    :return: filename for output gravity in validation
    """

    filename_output_gravity = dataset[idx]['filename']

    return filename_output_gravity
