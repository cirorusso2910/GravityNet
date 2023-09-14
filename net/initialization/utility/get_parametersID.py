def get_parametersID(experimentID):
    """ | ----------------- |
        | GET PARAMETERS ID |
        | ----------------- |

        get parameters ID from experiment ID

    """

    split_parameters = experimentID.split("|")
    num_parameters = len(split_parameters)

    experimentsID_parameters = experimentsID_parameters_dict()

    for i in range(num_parameters):
        param = split_parameters[i]
        param_name = param.split('=')[0]
        param_value = param.split('=')[1]

        if param_name == 'dataset':
            experimentsID_parameters['dataset'].append(param_value)

        elif param_name == 'split':
            experimentsID_parameters['split'].append(param_value)

        elif param_name == 'channel':
            experimentsID_parameters['channel'].append(param_value)

        elif param_name == 'norm':
            experimentsID_parameters['norm'].append(param_value)

        elif param_name == 'ep':
            experimentsID_parameters['ep'].append(param_value)

        elif param_name == 'lr':
            experimentsID_parameters['lr'].append(param_value)

        elif param_name == 'bs':
            experimentsID_parameters['bs'].append(param_value)

        elif param_name == 'backbone':
            experimentsID_parameters['backbone'].append(param_value)

        elif param_name == 'config':
            experimentsID_parameters['config'].append(param_value)

        elif param_name == 'hook':
            experimentsID_parameters['hook'].append(param_value)

        elif param_name == 'eval':
            experimentsID_parameters['eval'].append(param_value)

        elif param_name == 'GPU':
            experimentsID_parameters['GPU'].append(param_value)

    return experimentsID_parameters


def experimentsID_parameters_dict():
    """ experiments ID parameters dict """

    experiments_ID_dict = {
        'dataset': [],
        'split': [],
        'channel': [],
        'norm': [],
        'ep': [],
        'lr': [],
        'bs': [],
        'backbone': [],
        'config': [],
        'hook': [],
        'eval': [],
        'GPU': []
    }

    return experiments_ID_dict
