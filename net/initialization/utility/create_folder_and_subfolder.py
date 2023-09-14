import os


def create_folder_and_subfolder(main_path: str,
                                subfolder_path_dict: dict):
    """
    Create folder (main-path) and subfolder (subfolder-path-dict) of main-folder

    :param main_path: main path
    :param subfolder_path_dict: subfolder dictionary path
    """

    # create experiment result folder
    if not os.path.exists(main_path):
        os.mkdir(main_path)

    # for each path in experiment result path dict create folder
    for path in subfolder_path_dict:

        # for each sub path in experiment result path dict create sub folder
        if type(subfolder_path_dict[path]) is dict:
            for sub_path in subfolder_path_dict[path]:
                if not os.path.exists(subfolder_path_dict[path][sub_path]):
                    os.mkdir(subfolder_path_dict[path][sub_path])

        else:
            if not os.path.exists(subfolder_path_dict[path]):
                os.mkdir(subfolder_path_dict[path])