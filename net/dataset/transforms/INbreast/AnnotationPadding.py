import numpy as np


class AnnotationPadding(object):
    """
    Annotation padding
    """

    def __init__(self, max_padding: int):
        """
        __init__ method: run one when instantiating the object

        :param max_padding: max size of padding
        """

        self.max_padding = max_padding

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read sample
        annotation = sample['annotation']
        annotation_w48m14 = sample['annotation_w48m14']

        # add padding
        annotation_pad = np.ones((self.max_padding, annotation.shape[1])) * -1
        shape = np.shape(annotation)
        annotation_pad[:shape[0], :shape[1]] = annotation

        # add padding
        annotation_w48m14_pad = np.ones((self.max_padding, annotation_w48m14.shape[1])) * -1
        shape_w48m14 = np.shape(annotation_w48m14)
        annotation_w48m14_pad[:shape_w48m14[0], :shape_w48m14[1]] = annotation_w48m14

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'image_w48m14': sample['image_w48m14'],
                  'image_mask': sample['image_mask'],
                  'image_mask_w48m14': sample['image_mask_w48m14'],
                  'annotation': annotation_pad,
                  'annotation_w48m14': annotation_w48m14_pad
                  }

        return sample
