def timer(time_elapsed: float) -> dict:
    """
    Convert time elapsed (seconds) in: hours (h), minutes (m) and seconds (s)

    :param time_elapsed: time elapsed (seconds)
    :return: time h:m:s dictionary
    """

    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    # print("Time: {:.0f} h {:.0f} m {:.0f} s".format(int(hours), int(minutes), int(seconds)))

    time = {
        'hours': int(hours),
        'minutes': int(minutes),
        'seconds': int(seconds)
    }

    return time
