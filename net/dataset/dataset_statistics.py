from torch.utils.data import Dataset

from net.dataset.statistics.min_max_statistics import min_max_statistics
from net.dataset.statistics.standard_statistics import standard_statistics
from net.metrics.utility.my_round_value import my_round_value


def dataset_statistics(dataset_train: Dataset,
                       dataset_val: Dataset,
                       dataset_test: Dataset) -> dict:
    """
    Compute dataset statistics: (min, max) and (mean, std)

    :param dataset_train: dataset train
    :param dataset_val: dataset validation
    :param dataset_test: dataset test
    :return: dataset statistics dictionary
    """

    # min-max statistics
    min_train, max_train = min_max_statistics(dataset=dataset_train)
    print("Min-Max Statistics train: COMPLETE")
    min_val, max_val = min_max_statistics(dataset=dataset_val)
    print("Min-Max Statistics validation: COMPLETE")
    min_test, max_test = min_max_statistics(dataset=dataset_test)
    print("Min-Max Statistics test: COMPLETE")

    # std statistics
    mean_train, std_train = standard_statistics(dataset=dataset_train)
    print("Std Statistics train: COMPLETE")
    mean_val, std_val = standard_statistics(dataset=dataset_val)
    print("Std Statistics validation: COMPLETE")
    mean_test, std_test = standard_statistics(dataset=dataset_test)
    print("Std Statistics test: COMPLETE")

    print("\n-------------------"
          "\nDATASET STATISTICS:"
          "\n-------------------"
          "\nDATASET TRAIN:"
          "\nmin: {}  |  max: {}".format(min_train, max_train),
          "\nmean: {} |  std: {}".format(mean_train, std_train),
          "\nDATASET VALIDATION:"
          "\nmin: {}  |  max: {}".format(min_val, max_val),
          "\nmean: {} |  std: {}".format(mean_val, std_val),
          "\nDATASET TEST:"
          "\nmin: {}  |  max: {}".format(min_test, max_test),
          "\nmean: {} |  std: {}".format(mean_test, std_test))

    dataset_statistics_results = {
        'train': {
            'min': int(min_train),
            'max': int(max_train),
            'mean': my_round_value(mean_train, digits=3),
            'std': my_round_value(std_train, digits=3),
        },

        'validation': {
            'min': int(min_val),
            'max': int(max_val),
            'mean': my_round_value(mean_val, digits=3),
            'std': my_round_value(std_val, digits=3),
        },

        'test': {
            'min': int(min_test),
            'max': int(max_test),
            'mean': my_round_value(mean_test, digits=3),
            'std': my_round_value(std_test, digits=3),
        },
    }

    return dataset_statistics_results
