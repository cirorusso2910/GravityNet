from pandas import read_csv

from net.initialization.dict.metrics import metrics_dict
from net.initialization.header.metrics import metrics_header


def metrics_resume(metrics_resume_path: str) -> dict:
    """
    Resume metrics dictionary

    :param metrics_resume_path: metrics path from resume experiment
    :return: metrics dictionary resumed
    """

    header_metrics = metrics_header(metrics_type='train')
    metrics = metrics_dict(metrics_type='train')

    metrics_resume_csv = read_csv(metrics_resume_path, usecols=header_metrics, float_precision='round_trip')

    metrics['ticks'] = metrics_resume_csv["EPOCH"].tolist()
    metrics['loss']['loss'] = metrics_resume_csv["LOSS"].tolist()
    metrics['loss']['classification'] = metrics_resume_csv["CLASSIFICATION LOSS"].tolist()
    metrics['loss']['regression'] = metrics_resume_csv["REGRESSION LOSS"].tolist()
    metrics['learning_rate'] = metrics_resume_csv['LEARNING RATE'].tolist()
    metrics['AUC'] = metrics_resume_csv["AUC"].tolist()
    metrics['sensitivity']['work_point'] = metrics_resume_csv["SENSITIVITY WORK POINT"].tolist()
    metrics['sensitivity']['max'] = metrics_resume_csv["SENSITIVITY MAX"].tolist()
    metrics['AUFROC']['[0, 1]'] = metrics_resume_csv["AUFROC [0, 1]"].tolist()
    metrics['AUFROC']['[0, 10]'] = metrics_resume_csv["AUFROC [0, 10]"].tolist()
    metrics['AUFROC']['[0, 50]'] = metrics_resume_csv["AUFROC [0, 50]"].tolist()
    metrics['AUFROC']['[0, 100]'] = metrics_resume_csv["AUFROC [0, 100]"].tolist()
    metrics['time']['train'] = metrics_resume_csv["TIME TRAIN"].tolist()
    metrics['time']['validation'] = metrics_resume_csv["TIME VALIDATION"].tolist()
    metrics['time']['metrics'] = metrics_resume_csv["TIME METRICS"].tolist()

    return metrics
