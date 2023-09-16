def dataset_folders_dict() -> dict:
    """
    example of dataset folders dictionary
    :return: folders dictionary
    """

    dataset_folders = {
        'annotations': 'annotations',
        'annotations_subfolder': {
            'csv': 'csv',
            'draw': 'draw',
            'masks': 'masks'
        },

        'images': 'images',
        'images_subfolder': {
            'all': 'all',
            'masks': 'masks',
        },

        'info': 'info',
        'lists': 'lists',
        'split': 'split',
        'statistics': 'statistics',
    }

    return dataset_folders
