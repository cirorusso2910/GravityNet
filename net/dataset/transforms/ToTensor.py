import torch


class ToTensor(object):
    """
    Convert data sample to tensor
    """

    def __call__(self,
                 sample: dict) -> dict:
        """
        __call__ method: the instances behave like functions and can be called like a function.

        :param sample: sample dictionary
        :return: sample dictionary
        """

        # sample
        image = sample['image']
        image_mask = sample['image_mask']
        annotation = sample['annotation']

        image = torch.from_numpy(image).float()  # image to tensor
        image_mask = torch.from_numpy(image_mask.copy()).float()  # image mask to tensor
        annotation = torch.from_numpy(annotation).float()  # annotation to tensor

        sample = {'filename': sample['filename'],
                  'image': image,
                  'image_mask': image_mask,
                  'annotation': annotation
                  }

        return sample
