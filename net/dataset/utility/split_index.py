from typing import Tuple, List


def split_index(data_split: dict) -> Tuple[List, List, List]:
    """
    Get split index

    :param data_split: data split dictionary
    :return: train index list,
             validation index list,
             test index list
    """

    # num images
    num_images = len(data_split['index'])

    # init
    train_index = []
    validation_index = []
    test_index = []

    for i in range(num_images):

        # train index
        if data_split['split'][i] == 'train':
            train_index.append(data_split['index'][i])

        # validation index
        elif data_split['split'][i] == 'validation':
            validation_index.append(data_split['index'][i])

        # test index
        elif data_split['split'][i] == 'test':
            test_index.append(data_split['index'][i])

    return train_index, validation_index, test_index
