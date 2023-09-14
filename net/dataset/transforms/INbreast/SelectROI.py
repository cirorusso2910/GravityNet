import numpy as np


class SelectROI(object):
    """
    Select annotation according to ROI type
    """

    def __init__(self, roi_type):
        """
        __init__ method: run one when instantiating the object

        :param roi_type: ROI type
        """

        self.roi_type = roi_type

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # annotation
        annotation = sample['annotation']  # numpy.ndarray

        # num annotation
        num_annotation = len(annotation)

        # init annotation selected
        annotation_selected = []

        # for eah annotation
        for i in range(num_annotation):
            # if annotation-ROI is equal to ROI-type
            if annotation[i][0] == self.roi_type:
                # select annotation
                annotation_selected.append(annotation[i])

        # annotation selected to numpy
        annotation_selected = np.array(annotation_selected)  # numpy.ndarray

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'image_mask': sample['image_mask'],
                  'annotation': annotation_selected,
                  'annotation_w48m14': sample['annotation_w48m14']
                  }

        return sample
