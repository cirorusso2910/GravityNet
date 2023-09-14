def plot_title_dict(parser):
    """
    Define plot title experiment results

    :param parser: parser of parameters-parsing
    :return: plot title dictionary
    """

    plot_title = {
        'plots_train': {
            'loss': 'LOSS',
            'learning_rate': 'LEARNING RATE'
        },

        'plots_validation': {
            'sensitivity': 'SENSITIVITY',
            'AUC': 'AUC',
            'AUFROC': 'AUFROC'
        },

        'plots_test': {
            'FROC': 'FROC (TEST)',
            'ROC': 'ROC (TEST)',

            'score_distribution': "SCORE DISTRIBUTION (TEST)",

        },

        'plots_test_NMS': {
            'FROC': 'FROC (TEST NMS {}x{})'.format(parser.NMS_box_radius, parser.NMS_box_radius),
            'ROC': 'ROC (TEST NMS {}x{})'.format(parser.NMS_box_radius, parser.NMS_box_radius),

            'score_distribution': 'SCORE DISTRIBUTION (TEST NMS {}x{})'.format(parser.NMS_box_radius, parser.NMS_box_radius),
        }
    }

    return plot_title
