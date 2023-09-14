from net.plot.coords.read_coords import read_coords


def read_plots_test_coords(coords_test_path: dict) -> dict:
    """
    Read plots test coords

    :param coords_test_path: coords test path dictionary
    :return: coords dictionary
    """

    # FROC test
    FPS, sens = read_coords(coords_path=coords_test_path['FROC'],
                            coords_type='FROC')
    print("FROC coords reading: COMPLETE")

    # ROC test
    FPR, TPR = read_coords(coords_path=coords_test_path['ROC'],
                           coords_type='ROC')
    print("ROC coords reading: COMPLETE")

    coords = {
        'FROC': {
            'FPS': FPS,
            'sens': sens
        },

        'ROC': {
            'FPR': FPR,
            'TPR': TPR
        }
    }

    return coords
