import numpy as np


class Annotation(object):
    """
    Annotation definition
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
        annotation_w48m14 = sample['annotation_w48m14']

        # read image
        image = sample['image']
        image_w48m14 = sample['image_w48m14']

        # num annotation
        num_annotation = len(annotation)
        num_annotation_w48m14 = len(annotation_w48m14)

        # init
        annotation_np = np.zeros((0, 4))
        annotation_w48m14_np = np.zeros((0, 3))

        # annotation definition
        for i in range(num_annotation):
            x = int(annotation[i][1])
            y = int(annotation[i][2])
            radius = int(annotation[i][3])
            intensity = image[y, x]

            line_np = np.array((x, y, radius, intensity))

            annotation_np = np.append(annotation_np, [line_np], axis=0)

        # annotation w48m14 definition (only for training)
        for i in range(num_annotation_w48m14):
            x = int(annotation_w48m14[i][1])
            y = int(annotation_w48m14[i][2])
            intensity = image_w48m14[y, x]

            line_np = np.array((x, y, intensity))

            annotation_w48m14_np = np.append(annotation_w48m14_np, [line_np], axis=0)

        sample = {'filename': sample['filename'],
                  'image': sample['image'],
                  'image_w48m14': sample['image_w48m14'],
                  'image_mask': sample['image_mask'],
                  'image_mask_w48m14': sample['image_mask_w48m14'],
                  'annotation': annotation_np,
                  'annotation_w48m14': annotation_w48m14_np
                  }

        return sample
