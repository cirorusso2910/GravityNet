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

        }
    }

    return plot_title
