from net.metrics.utility.my_round_value import my_round_value


def box_radius_rescale(box_draw_radius: int,
                       rescale: float) -> float:
    """
    Box radius rescale according to rescale factor

    :param box_draw_radius: box draw radius (mean radius 4 + 1 to be odd)
    :param rescale: rescale factor
    :return: box radius rescaled
    """

    # box draw radius rescaled
    box_draw_radius_rescaled = box_draw_radius * rescale
    box_draw_radius_rescaled = my_round_value(value=box_draw_radius_rescaled, digits=3)

    return box_draw_radius_rescaled
