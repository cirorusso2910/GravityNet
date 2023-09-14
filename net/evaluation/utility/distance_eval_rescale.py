from net.metrics.utility.my_round_value import my_round_value


def distance_eval_rescale(eval: str,
                          rescale: float) -> float:
    """
    Rescale distance for evaluation according to rescale factor

    :param eval: eval
    :param rescale: rescale factor
    :return: distance rescaled
    """

    # eval with rescale
    distance = float(eval.split('e')[1])

    # distance rescaled
    distance_rescaled = distance * rescale
    distance_rescaled = my_round_value(value=distance_rescaled, digits=3)

    return distance_rescaled
