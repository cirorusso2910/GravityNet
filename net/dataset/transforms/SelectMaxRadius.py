import numpy as np


class SelectMaxRadius(object):
    """
    Select max annotation radius between radius in x and radius in y
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read annotation
        annotation = sample['annotation']
        num_annotations = annotation.shape[0]

        # annotation with radius
        annotation_radius = np.zeros((annotation.shape[0], 3))
        annotation_radius[:, 0] = annotation[:, 0]
        annotation_radius[:, 1] = annotation[:, 1]

        for i in range(num_annotations):

            radius_x = annotation[i, 2]  # radius in x
            radius_y = annotation[i, 3]  # radius in y

            if radius_x > radius_y:
                annotation_radius[i, 2] = radius_x
            else:
                annotation_radius[i, 2] = radius_y

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'image_mask': sample['image_mask'],
                  'annotation': annotation_radius
                  }

        return sample
