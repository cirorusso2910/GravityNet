def msg_load_best_model_complete(metrics_type: str,
                                 load_model: dict):
    """
    Message load best model complete

    :param metrics_type: metrics type
    :param load_model: load model dictionary
    """

    print("LOADED BEST MODEL ({}):".format(metrics_type.upper()),
          "\ntrained for {} epochs with {}: {:.3f}".format(load_model['epoch'],
                                                           metrics_type,
                                                           load_model[metrics_type]))
