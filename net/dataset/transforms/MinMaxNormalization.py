from torchvision import transforms


class MinMaxNormalization(object):
    """
    Min-Max Normalization: (usually called Feature Scaling) performs a linear transformation on the original data.
    This technique gets all the scaled data in the range (0, 1)

                       (x - x_min)
        x_scaled  =  ---------------                    for each channel
                     (x_max - x_min)

    min-max normalization preserves the relationships among the original data values

    """

    def __init__(self,
                 min: int,
                 max: int):
        """
        __init__ method: run one when instantiating the object

        :param min: min value
        :param max: max value
        """

        self.min = min
        self.max = max

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read sample
        image = sample['image']

        # min max normalization (image range to [0, 1])
        transform = transforms.Normalize([self.min, self.min, self.min], [self.max - self.min, self.max - self.min, self.max - self.min])
        image_normalized = transform(image)

        sample = {'filename': sample['filename'],
                  'image': image_normalized,
                  'image_mask': sample['image_mask'],
                  'annotation': sample['annotation']
                  }

        return sample
