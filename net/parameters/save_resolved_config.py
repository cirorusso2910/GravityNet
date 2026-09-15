import argparse
import os

import yaml


def save_resolved_config(parser: argparse.Namespace,
                         experiment_path: str):
    """
    Save the fully-resolved parameters (CLI + YAML + defaults) as YAML in the experiment
    folder, so a past run's exact configuration can always be inspected or reproduced with --cfg.

    :param parser: parser of parameters-parsing
    :param experiment_path: experiment folder path
    """

    config_path = os.path.join(experiment_path, "config.yaml")

    with open(config_path, 'w') as file:
        yaml.safe_dump(vars(parser), file, sort_keys=False)
