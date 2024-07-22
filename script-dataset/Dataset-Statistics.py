import csv

from torchvision.transforms import transforms

from net.dataset.classes.dataset_class import dataset_class
from net.dataset.dataset_split import dataset_split
from net.dataset.dataset_statistics import dataset_statistics
from net.dataset.statistics.num_annotations import get_num_annotations
from net.dataset.statistics.num_normal_images import get_num_normal_images
from net.dataset.transforms.AnnotationPadding import AnnotationPadding
from net.dataset.transforms.Rescale import Rescale
from net.dataset.transforms.ToTensor import ToTensor
from net.initialization.ID.experimentID import experimentID
from net.initialization.header.statistics import statistics_header
from net.initialization.init import initialization
from net.parameters.parameters import parameters_parsing
from net.utility.execution_mode import execution_mode
from net.utility.msg.msg_load_dataset_complete import msg_load_dataset_complete
from net.utility.read_split import read_split


def main():
    """
        | ------------------ |
        | DATASET STATISTICS |
        | ------------------ |

        Compute dataset statistics

    """

    print("| ------------------ |\n"
          "| DATASET STATISTICS |\n"
          "| ------------------ |\n")

    # ------------------ #
    # PARAMETERS-PARSING #
    # ------------------ #
    # command line parameter parsing
    parser = parameters_parsing()

    # execution mode start
    execution_mode(mode=parser.mode,
                   option='start')

    # -------------- #
    # INITIALIZATION #
    # -------------- #
    print("\n---------------"
          "\nINITIALIZATION:"
          "\n---------------")
    # experiment ID
    experiment_ID = experimentID(typeID=parser.typeID,
                                 sep=parser.sep,
                                 parser=parser)

    # initialization
    path = initialization(network_name="GravityNet",
                          experiment_ID=experiment_ID,
                          parser=parser)

    # read data split
    data_split = read_split(path_split=path['dataset']['split'])

    # ------------ #
    # LOAD DATASET #
    # ------------ #
    print("\n-------------"
          "\nLOAD DATASET:"
          "\n-------------")
    dataset = dataset_class(images_dir=path['dataset']['images']['all'],
                            images_extension=parser.images_extension,
                            images_masks_dir=path['dataset']['images']['masks'],
                            images_masks_extension=parser.images_masks_extension,
                            annotations_dir=path['dataset']['annotations']['all'],
                            annotations_extension=parser.annotations_extension,
                            filename_list=data_split['filename'],
                            transforms=transforms.Compose([
                                # DATA
                                Rescale(rescale=parser.rescale,
                                        num_channels=parser.num_channels),  # Rescale images and annotations
                                # DATA PREPARATION
                                AnnotationPadding(max_padding=parser.max_padding),  # annotation padding (for batch dataloader)
                                ToTensor(),  # To Tensor
                            ]))

    msg_load_dataset_complete(dataset_name=parser.dataset)

    # ------------- #
    # DATASET SPLIT #
    # ------------- #
    # subset dataset according to data split
    dataset_train, dataset_val, dataset_test = dataset_split(data_split=data_split,
                                                             dataset=dataset)

    # ------------------ #
    # DATASET NUM IMAGES #
    # ------------------ #
    # num images for dataset-train, dataset-val, dataset-test
    num_images_train = dataset_train.__len__()
    num_images_val = dataset_val.__len__()
    num_images_test = dataset_test.__len__()

    # num images dict
    num_images = {
        'train': num_images_train,
        'validation': num_images_val,
        'test': num_images_test
    }

    # -------------------------- #
    # DATASET NUM IMAGES NORMALS #
    # -------------------------- #
    # num normal images for dataset-train, dataset-val, dataset-test
    num_normal_images_train = get_num_normal_images(dataset=dataset_train)
    num_normal_images_val = get_num_normal_images(dataset=dataset_val)
    num_normal_images_test = get_num_normal_images(dataset=dataset_test)

    # num normal images dict
    num_normal_images = {
        'train': num_normal_images_train,
        'validation': num_normal_images_val,
        'test': num_normal_images_test,
    }

    # ----------------------- #
    # DATASET NUM ANNOTATIONS #
    # ----------------------- #
    # num annotations for dataset-train, dataset-val, dataset-test
    num_annotations_train = get_num_annotations(dataset=dataset_train)
    num_annotations_val = get_num_annotations(dataset=dataset_val)
    num_annotations_test = get_num_annotations(dataset=dataset_test)

    # num annotations dict
    num_annotations = {
        'train': num_annotations_train,
        'validation': num_annotations_val,
        'test': num_annotations_test,
    }

    # ------------------ #
    # DATASET STATISTICS #
    # ------------------ #
    print("\n-------------------"
          "\nDATASET STATISTICS:"
          "\n-------------------")
    dataset_statistics_dict = dataset_statistics(dataset_train=dataset_train,
                                                 dataset_val=dataset_val,
                                                 dataset_test=dataset_test)

    # dataset-statistics.csv
    dataset_statistics_csv(statistics_path=path['dataset']['statistics'],
                           num_images=num_images,
                           num_normal_images=num_normal_images,
                           num_annotations=num_annotations,
                           small_lesion=parser.small_lesion,
                           dataset_statistics_dict=dataset_statistics_dict)


def dataset_statistics_csv(statistics_path: str,
                           num_images: dict,
                           num_normal_images: dict,
                           num_annotations: dict,
                           small_lesion: str,
                           dataset_statistics_dict: dict):
    """
    save dataset-statistics.csv

    :param statistics_path: path to save
    :param num_images: num images
    :param num_annotations: num annotations
    :param dataset_statistics_dict: dataset statistics dictionary
    """

    with open(statistics_path, 'w') as file:
        # writer
        writer = csv.writer(file)

        # write header
        header = statistics_header(statistics_type='statistics',
                                   small_lesion_type=small_lesion.upper())
        writer.writerow(header)

        # write row
        writer.writerow(["TRAIN",
                         num_images['train'],
                         num_normal_images['train'],
                         num_annotations['train'],
                         dataset_statistics_dict['train']['min'],
                         dataset_statistics_dict['train']['max'],
                         dataset_statistics_dict['train']['mean'],
                         dataset_statistics_dict['train']['std']
                         ])

        # write row
        writer.writerow(["VALIDATION",
                         num_images['validation'],
                         num_normal_images['validation'],
                         num_annotations['validation'],
                         dataset_statistics_dict['validation']['min'],
                         dataset_statistics_dict['validation']['max'],
                         dataset_statistics_dict['validation']['mean'],
                         dataset_statistics_dict['validation']['std']
                         ])

        # write row
        writer.writerow(["TEST",
                         num_images['test'],
                         num_normal_images['test'],
                         num_annotations['test'],
                         dataset_statistics_dict['test']['min'],
                         dataset_statistics_dict['test']['max'],
                         dataset_statistics_dict['test']['mean'],
                         dataset_statistics_dict['test']['std']
                         ])


if __name__ == '__main__':
    main()
