from torchvision import transforms


class StandardNormalization(object):
    """
    Standard Normalization: Standardization or Z-Score Normalization is the transformation of features
    by subtracting from mean and dividing by standard deviation

                  (image - mean)
        image =  ----------------               for each channel
                       std

    """

    def __init__(self,
                 mean: float,
                 std: float):
        """
        __init__ method: run one when instantiating the object

        :param mean: mean
        :param std: std
        """

        self.mean = mean
        self.std = std

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # read sample
        image = sample['image']

        # standard normalization
        transform = transforms.Normalize([self.mean, self.mean, self.mean], [self.std, self.std, self.std])
        image_normalized = transform(image)

        sample = {'filename': sample['filename'],
                  'image': image_normalized,
                  'image_mask': sample['image_mask'],
                  'annotation': sample['annotation']
                  }

        return sample
