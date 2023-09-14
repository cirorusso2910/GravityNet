import torch


def get_single_detection(index: int,
                         detections: torch.Tensor) -> dict:
    """
    Get single detection for specific index

    :param index: index
    :param detections: detections
    :return: single detection dictionary
    """

    label = int(detections[index, 1].item())
    score = float(detections[index, 2].item())
    prediction_coord_x = int(round(detections[index, 3].item(), ndigits=0))
    prediction_coord_y = int(round(detections[index, 4].item(), ndigits=0))

    # annotation hooked only if prediction is TP (label '1')
    if label == 1.0:
        annotation_hooked_coord_x = int(round(detections[index, 5].item(), ndigits=3))
        annotation_hooked_coord_y = int(round(detections[index, 6].item(), ndigits=3))
    else:
        annotation_hooked_coord_x = detections[index, 5]
        annotation_hooked_coord_y = detections[index, 6]

    single_detection = {
        'label': label,
        'score': score,
        'prediction_x': prediction_coord_x,
        'prediction_y': prediction_coord_y,
        'annotation_x': annotation_hooked_coord_x,
        'annotation_y': annotation_hooked_coord_y
    }

    return single_detection
