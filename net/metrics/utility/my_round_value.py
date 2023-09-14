from cmath import isnan


def my_round_value(value: float,
                   digits: int) -> float:
    """
    My round value
    - if value is 'nan': set ''

    :param value: value
    :param digits: number of digits
    :return: value rounded
    """

    # round value
    value_rounded = round(value, ndigits=digits)

    # replace nan value whit '' in metrics.csv
    if isnan(value_rounded):
        value_rounded = ''

    return value_rounded
