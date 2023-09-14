def INbreast_dataset_folders_dict() -> dict:
    """
    INbreast dataset folders dictionary
    :return: folders dictionary
    """

    dataset_folders = {
        'annotations': 'annotations',
        'annotations_subfolder': {
            'csv': 'csv',
            'csv_subfolder': {
                'all': 'all',
                'asymmetries': 'asymmetries',
                'calcifications': 'calcifications',
                'calcifications_cropped': 'calcifications-cropped',
                'calcifications_cropped_filtered': 'calcifications-cropped-filtered',
                'calcifications_filtered': 'calcifications-filtered',
                'calcifications_w48m14': 'calcifications-w48m14',
                'calcifications_w48m14_cropped': 'calcifications-w48m14-cropped',
                'clusters': 'clusters',
                'masses': 'masses',
                'normals': 'normals',
                'spiculated_regions': 'spiculated-regions',
            },

            'draw': 'draw',
            'draw_subfolder': {
                'all': 'all',
                'asymmetries': 'asymmetries',
                'calcifications': 'calcifications',
                'calcifications_cropped': 'calcifications-cropped',
                'calcifications_cropped_filtered': 'calcifications-cropped-filtered',
                'calcifications_filtered': 'calcifications-filtered',
                'calcifications_w48m14': 'calcifications-w48m14',
                'calcifications_w48m14_cropped': 'calcifications-w48m14-cropped',
                'clusters': 'clusters',
                'masses': 'masses',
                'normals': 'normals',
                'spiculated_regions': 'spiculated-regions',
            },

            'masks': 'masks',
            'PROI': 'PROI',
            'roi': 'roi',
            'xml': 'xml',
        },

        'images': 'images',
        'images_subfolder': {
            'all': 'all',
            'all_cropped': 'all-cropped',
            'all_w48m14_cropped': 'all-w48m14-cropped',
            'masks': 'masks',
            'masks_cropped': 'masks-cropped',
            'masks_w48m14_cropped': 'masks-w48m14-cropped',
        },

        'info': 'info',
        'lists': 'lists',
        'split': 'split',
        'statistics': 'statistics',
    }

    return dataset_folders
