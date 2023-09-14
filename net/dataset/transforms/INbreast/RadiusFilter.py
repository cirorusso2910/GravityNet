import numpy as np


class RadiusFilter(object):
    """
    Delete annotation with radius greater than a defined value
    """

    def __init__(self, radius: int):
        """
        __init__ method: run one when instantiating the object

        :param radius: radius bound (default: 7)
        """

        self.radius = radius

    def __call__(self, sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # annotation
        annotation = sample['annotation']

        if len(annotation) > 0:
            # radius
            radius = annotation[:, 3]

            # delete calcifications with radius greater than a defined value
            radius_filtered_index = np.squeeze(np.argwhere(radius < self.radius), axis=1)
            annotation_filtered = annotation[radius_filtered_index]

        else:
            annotation_filtered = annotation

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'image_w48m14': sample['image_w48m14'],
                  'image_mask': sample['image_mask'],
                  'image_mask_w48m14': sample['image_mask_w48m14'],
                  'annotation': annotation_filtered,
                  'annotation_w48m14': sample['annotation_w48m14']
                  }

        return sample
