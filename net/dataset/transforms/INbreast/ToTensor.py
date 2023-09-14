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

        image, image_mask, annotation = sample['image'], sample['image_mask'], sample['annotation']
        image_w48m14, image_mask_w48m14, annotation_w48m14 = sample['image_w48m14'], sample['image_mask_w48m14'], sample['annotation_w48m14']

        image = torch.from_numpy(image).float()
        image_mask = torch.from_numpy(image_mask.copy()).float()
        annotation = torch.from_numpy(annotation).float()

        image_w48m14 = torch.from_numpy(image_w48m14).float()
        image_mask_w48m14 = torch.from_numpy(image_mask_w48m14.copy()).float()
        annotation_w48m14 = torch.from_numpy(annotation_w48m14).float()

        sample = {'filename': sample['filename'],
                  'image': image,
                  'image_w48m14': image_w48m14,
                  'image_mask': image_mask,
                  'image_mask_w48m14': image_mask_w48m14,
                  'annotation': annotation,
                  'annotation_w48m14': annotation_w48m14
                  }

        return sample
