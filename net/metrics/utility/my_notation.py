def decimal_notation(number: str) -> float:
    """
    Transforms a real number with scientific notation in decimal notation
    example: 1e-04 -> 0.0001

    :param number: number in scientific notation
    :return: number in decimal notation
    """

    number_decimal_notation = int(number.split('e')[0]) / (10 ** -int(number.split('e')[1]))

    return number_decimal_notation


def scientific_notation(number):
    """
    Transforms a real number with a scientific notation
    example: 0.0001 -> 1e-04

    :param number: real number
    :return: number in scientific notation
    """

    format_number = '%e' % number
    number_scientific_notation = format_number.split('e')[0].rstrip('0').rstrip('.') + 'e' + format_number.split('e')[1]

    return number_scientific_notation
