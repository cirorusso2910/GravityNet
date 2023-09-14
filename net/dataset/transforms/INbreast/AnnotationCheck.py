import numpy as np


class AnnotationCheck(object):
    """
    AnnotationCheck: check annotation shape
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # sample annotation
        annotation = sample['annotation']  # numpy.ndarray
        annotation_w48m14 = sample['annotation_w48m14']  # numpy.ndarray

        # check annotation
        annotation_check = sample['annotation'].copy()
        num_annotation = len(annotation_check)
        if num_annotation == 0:
            annotation_check = np.zeros((0, 4))
        else:
            annotation_check = annotation

        # check annotation w48m14
        annotation_w48m14_check = annotation_w48m14.copy()
        num_annotation_w48m14 = len(annotation_w48m14_check)
        if num_annotation_w48m14 == 0:
            annotation_w48m14_check = np.zeros((0, 3))
        else:
            annotation_w48m14_check = annotation_w48m14

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'image_mask': sample['image_mask'],
                  'annotation': annotation_check,
                  'annotation_w48m14': annotation_w48m14_check
                  }

        return sample
