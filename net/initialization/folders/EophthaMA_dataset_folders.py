def EophthaMA_dataset_folders_dict() -> dict:
    """
    E-ophtha-MA dataset folders dictionary
    :return: folders dictionary
    """

    dataset_folders = {
        'images': 'images',
        'images_subfolder': {
            'all': 'all',
            'contours': 'contours',
            'cropped': 'cropped',
            'green': 'green',
            'masks': 'masks',
            'masks_subfolder': {
                'all': 'all',
                'cropped': 'cropped',
            },
            'resized': 'resized',
        },

        'annotations': 'annotations',
        'annotations_subfolder': {
            'csv': 'csv',
            'csv_subfolder': {
                'all': 'all',
                'cropped': 'cropped',
                'resized': 'resized',
            },
            'masks': 'masks',
            'draw': 'draw',
            'draw_subfolder': {
                'all': 'all',
                'cropped': 'cropped',
                'resized': 'resized',
            }
        },

        'info': 'info',
        'lists': 'lists',
        'split': 'split',
        'statistics': 'statistics'
    }

    return dataset_folders
