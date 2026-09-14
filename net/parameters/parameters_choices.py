# ------------------- #
# PARAMETERS CHOICHES #
# ------------------- #

parameters_choices = {
    'num_channels': [1, 3],
    'backbone': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152',
                'ResNeXt-50_32x4d', 'ResNeXt-101_32x8d', 'ResNeXt-101_64x4d',
                'DenseNet-121', 'DenseNet-161', 'DenseNet-169', 'DenseNet-201',
                'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3',
                'EfficientNet-B4', 'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7',
                'EfficientNetV2-S', 'EfficientNetV2-M', 'EfficientNetV2-L',
                'Swin-T', 'Swin-S', 'Swin-B'],
    'optimizer': ['Adam', 'SGD'],
    'scheduler': ['ReduceLROnPlateau', 'StepLR', 'CosineAnnealing'],
    'type_draw': ['circle', 'box']
}
