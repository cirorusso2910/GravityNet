from torch.utils.data import Dataset

from net.dataset.utility.viewable_image import viewable_image


def read_dataset_sample(dataset: Dataset,
                        idx: int) -> dict:
    """
    Read dataset sample at specific index (idx) position

    :param dataset: dataset
    :param idx: index
    :return: sample dictionary
    """

    # image filename
    image_filename = dataset[idx]['filename']

    # image
    image = dataset[idx]['image']
    image = image.permute((1, 2, 0))  # permute
    image = image.cpu().detach().numpy()
    image = viewable_image(image=image)

    # image mask
    image_mask = dataset[idx]['image_mask']
    image_mask = image_mask.cpu().detach().numpy()

    # read annotation
    annotation = dataset[idx]['annotation']
    annotation = annotation[annotation[:, 0] != -1]  # delete padding

    sample = {
        'filename': image_filename,
        'image': image,
        'image_mask': image_mask,
        'annotation': annotation
    }

    return sample
