# ------------------- #
# PARAMETERS CHOICHES #
# ------------------- #

parameters_choices = {
    'num_channels': [1, 3],
    'backbone': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
    'optimizer': ['Adam', 'SGD'],
    'scheduler': ['ReduceLROnPlateau', 'StepLR', 'CosineAnnealing'],
    'type_draw': ['circle', 'box']
}
