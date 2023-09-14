def radius_eval(eval: str) -> int:
    """
    Get radius factor from eval parameter

    example: radius1 -> radius with factor 1

    :param eval: eval
    :return: radius factor
    """

    # radius factor
    radius = int(eval.split('s')[1])

    return radius
