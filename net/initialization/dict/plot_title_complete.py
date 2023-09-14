def plot_title_complete_dict():
    """
    Define plot title experiment complete results

    :return: plot title dictionary
    """

    plot_title = {
        'plots_test': {
            'FROC': 'FROC (EXPERIMENT COMPLETE)',
            'ROC': 'FROC (EXPERIMENT COMPLETE)',

            'score_distribution': 'SCORE DISTRIBUTION (EXPERIMENT COMPLETE)',
        },

        'plots_test_NMS': {
            '1x1': {
                'FROC': 'FROC (EXPERIMENT COMPLETE - NMS 1x1)',
                'ROC': 'FROC (EXPERIMENT COMPLETE - NMS 1x1)',

                'score_distribution': 'SCORE DISTRIBUTION (EXPERIMENT COMPLETE - NMS 1x1)',
            },

            '3x3': {
                'FROC': 'FROC (EXPERIMENT COMPLETE - NMS 3x3)',
                'ROC': 'FROC (EXPERIMENT COMPLETE - NMS 3x3)',

                'score_distribution': 'SCORE DISTRIBUTION (EXPERIMENT COMPLETE - NMS 3x3)',
            },

            '5x5': {
                'FROC': 'FROC (EXPERIMENT COMPLETE - NMS 5x5)',
                'ROC': 'FROC (EXPERIMENT COMPLETE - NMS 5x5)',

                'score_distribution': 'SCORE DISTRIBUTION (EXPERIMENT COMPLETE - NMS 5x5)',
            },

            '7x7': {
                'FROC': 'FROC (EXPERIMENT COMPLETE - NMS 7x7)',
                'ROC': 'FROC (EXPERIMENT COMPLETE - NMS 7x7)',

                'score_distribution': 'SCORE DISTRIBUTION (EXPERIMENT COMPLETE - NMS 7x7)',
            }
        }
    }

    return plot_title
