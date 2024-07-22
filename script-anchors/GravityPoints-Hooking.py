import os
import sys

import numpy as np
import torch
from torchvision.transforms import transforms

from net.anchors.gravity_points_config import gravity_points_config
from net.anchors.utility.check_image_shape import check_image_shape
from net.dataset.classes.dataset_class import dataset_class
from net.dataset.transforms.Add3ChannelsImage import Add3ChannelsImage
from net.dataset.transforms.AnnotationPadding import AnnotationPadding
from net.dataset.transforms.Rescale import Rescale
from net.dataset.transforms.ToTensor import ToTensor
from net.dataset.utility.read_dataset_sample import read_dataset_sample
from net.debug.debug_hooking import debug_hooking
from net.initialization.ID.experimentID import experimentID
from net.initialization.init import initialization
from net.initialization.utility.create_folder import create_folder
from net.parameters.parameters import parameters_parsing
from net.utility.execution_mode import execution_mode
from net.utility.msg.msg_load_dataset_complete import msg_load_dataset_complete
from net.utility.read_file import read_file


def main():
    """
        | ------------------------ |
        | GRAVITY POINTS - HOOKING |
        | ------------------------ |

        Show gravity-points hooking process details and save it on image (debug option)

    """

    print("| ======================== |\n"
          "| GRAVITY POINTS - HOOKING |\n"
          "| ======================== |\n")

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

    # filename list
    filename_list = read_file(file_path=path['dataset']['lists']['all'])

    # ------ #
    # DEVICE #
    # ------ #
    print("\n-------"
          "\nDEVICE:"
          "\n-------")
    device = torch.device("cpu")  # device cpu
    print("Device: {}".format(device))

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
                            filename_list=filename_list,
                            transforms=transforms.Compose([
                                # DATA
                                Rescale(rescale=parser.rescale,
                                        num_channels=parser.num_channels),  # Rescale all images and annotations
                                # DATA PREPARATION
                                Add3ChannelsImage(num_channels=parser.num_channels),  # Add 3 Channels to image [C, H, W]
                                AnnotationPadding(max_padding=parser.max_padding),  # annotations padding (for batch dataloader)
                                ToTensor()  # To Tensor
                            ]))

    msg_load_dataset_complete(dataset_name=parser.dataset)

    # -------------- #
    # GRAVITY POINTS #
    # -------------- #
    print("\n--------------"
          "\nGRAVITY POINTS:"
          "\n--------------")
    # image shape (C x H x W)
    image_channels, image_height, image_width = dataset[0]['image'].shape  # get image shape
    image_shape = np.array((int(image_height), int(image_width)))  # converts to numpy.array

    # check image shape dimension
    check_image_shape(image_shape=image_shape)

    # generate gravity points
    gravity_points, gravity_points_feature_map, feature_map_shape = gravity_points_config(config=parser.config,
                                                                                          image_shape=image_shape,
                                                                                          device=device)

    num_gravity_points = gravity_points.shape[0]  # num gravity points (A)

    # ------- #
    # HOOKING #
    # ------- #
    print("\n--------"
          "\nHOOKING:"
          "\n--------")
    # annotation
    sample = read_dataset_sample(dataset=dataset,
                                 idx=parser.idx)

    print("\nSAMPLE:"
          "\nFilename: {}".format(sample['filename']),
          '\nNum Nuclei: {}'.format(len(sample['annotation'])))

    annotation = sample['annotation']
    num_annotations = annotation.shape[0]

    if num_annotations == 0:
        sys.exit("\nNO NUCLEI!")

    dist = torch.cdist(gravity_points.float(), annotation[:, :2].float(), p=2)  # num_gravity_points (A) x num_annotations (T)
    dist_min, index_min = torch.min(dist, dim=1)  # A x 1

    # POSITIVE INDICES: gravity points with dist min <= hook dist
    positive_indices = torch.le(dist_min, parser.hook)

    # NEGATIVE INDICES: gravity points with dist min > hook dist + hook gap
    negative_indices = torch.gt(dist_min, parser.hook + parser.gap)

    # REJECTED INDICES: gravity points with dist min < hook dist and > hook gap
    rejected_indices = torch.logical_and(torch.gt(dist_min, parser.hook), torch.le(dist_min, parser.hook + parser.gap))

    # num positive gravity points
    num_positive_gravity_points = positive_indices.sum()

    # num negative gravity points
    num_negative_gravity_points = negative_indices.sum()

    # num rejected gravity points
    num_rejected_gravity_points = rejected_indices.sum()

    # annotation closest to each gravity points (with index min)
    assigned_annotations = annotation[index_min, :2]

    print("\nConfig: {}".format(parser.config),
          "\nGravity Points: {}".format(num_gravity_points),
          "\nHook: {}".format(parser.hook),
          "\n",
          "\nPositive Gravity Points: {} / {} ({:.3f} %)".format(num_positive_gravity_points, num_gravity_points, num_positive_gravity_points / num_gravity_points * 100, digits=3),
          "\nNegative Gravity Points: {} / {} ({:.3f} %)".format(num_negative_gravity_points, num_gravity_points, num_negative_gravity_points / num_gravity_points * 100, digits=3),
          "\nRejected Gravity Points: {} / {} ({:.3f} %)".format(num_rejected_gravity_points, num_gravity_points, num_rejected_gravity_points / num_gravity_points * 100, digits=3),
          "\n",
          "\nAnnotations: {}".format(num_annotations),
          "\nHooked Annotations: {} / {}".format(len(torch.unique(assigned_annotations[:, 0])), num_annotations))

    # init result folder
    result_folder_filename = "GravityPoints-Hooking|config={}".format(parser.config)
    doc_folder = "./doc/gravity-points-hooking"
    result_folder_path = os.path.join(doc_folder, result_folder_filename)
    create_folder(path=result_folder_path)

    # init result path
    hooking_image_filename = "GravityPoints-Hooking|filename={}|config={}|hook={}.png".format(sample['filename'], parser.config, parser.hook)
    hooking_image_path = os.path.join(result_folder_path, hooking_image_filename)

    # debug hooking
    debug_hooking(gravity_points=gravity_points.numpy(),
                  annotation=annotation,
                  assigned_annotations=assigned_annotations,
                  positive_indices=positive_indices,
                  negative_indices=negative_indices,
                  rejected_indices=rejected_indices,
                  image=sample['image'],
                  save=True,
                  path=hooking_image_path)


if __name__ == '__main__':
    main()
