import numpy as np


class AnnotationPadding(object):
    """
    Annotation padding
    """

    def __init__(self,
                 max_padding: int):
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

        # read annotation
        annotation = sample['annotation']

        # annotation padding
        annotation_pad = np.ones((self.max_padding, annotation.shape[1])) * -1
        shape = np.shape(annotation)
        annotation_pad[:shape[0], :shape[1]] = annotation

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'image_mask': sample['image_mask'],
                  'annotation': annotation_pad
                  }

        return sample
