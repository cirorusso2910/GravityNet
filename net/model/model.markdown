# NET.MODEL
**net.model** package

## STRUCTURE

    model
    |
    | - backbone
    |   | - DenseNet
    |   |   | MyDenseNet121.py
    |   |   | MyDenseNet161.py
    |   |   | MyDenseNet169.py
    |   |   | MyDenseNet201.py
    |   |
    |   | - EfficientNet
    |   |   | MyEfficientNetB0.py
    |   |   | MyEfficientNetB1.py
    |   |   | MyEfficientNetB2.py
    |   |   | MyEfficientNetB3.py
    |   |   | MyEfficientNetB4.py
    |   |   | MyEfficientNetB5.py
    |   |   | MyEfficientNetB6.py
    |   |   | MyEfficientNetB7.py
    |   |
    |   | - EfficientNetV2
    |   |   | MyEfficientNetV2L.py
    |   |   | MyEfficientNetV2M.py
    |   |   | MyEfficientNetV2S.py
    |   |
    |   | - ResNet
    |   |   | MyResNet18.py
    |   |   | MyResNet34.py
    |   |   | MyResNet50.py
    |   |   | MyResNet101.py
    |   |   | MyResNet152.py
    |   |
    |   | - ResNeXt
    |   |   | MyResNeXt50_32x4d.py
    |   |   | MyResNeXt101_32x8d.py
    |   |   | MyResNeXt101_64x4d.py
    |   |
    |   | - Swin
    |   |   | MySwinB.py
    |   |   | MySwinS.py
    |   |   | MySwinT.py
    |   |
    |   | MyDenseNet_models.py
    |   | MyEfficientNet_models.py
    |   | MyEfficientNetV2_models.py
    |   | MyResNet_models.py
    |   | MyResNeXt_models.py
    |   | MySwin_models.py
    |   
    | - gravitynet
    |   | ClassificationSubNet.py
    |   | GravityNet.py
    |   | RegressionSubNet.py
    |
    | - utility
        | load_model.py
        | my_torchsummary.py
        | save_model.py
    

## DOCUMENTATION

| FOLDER         | SUB-FOLDER     | FUNCTION                   | DESCRIPTION                          |
|----------------|----------------|----------------------------|--------------------------------------|
| **backbone**   | DenseNet       | MyDenseNet121.py           | My DenseNet121 implementation        |
| **backbone**   | DenseNet       | MyDenseNet161.py           | My DenseNet161 implementation        |
| **backbone**   | DenseNet       | MyDenseNet169.py           | My DenseNet169 implementation        |
| **backbone**   | DenseNet       | MyDenseNet201.py           | My DenseNet201 implementation        |
| **backbone**   | EfficientNet   | MyEfficientNetB0.py        | My EfficientNetB0 implementation     |
| **backbone**   | EfficientNet   | MyEfficientNetB1.py        | My EfficientNetB1 implementation     |
| **backbone**   | EfficientNet   | MyEfficientNetB2.py        | My EfficientNetB2 implementation     |
| **backbone**   | EfficientNet   | MyEfficientNetB3.py        | My EfficientNetB3 implementation     |
| **backbone**   | EfficientNet   | MyEfficientNetB4.py        | My EfficientNetB4 implementation     |
| **backbone**   | EfficientNet   | MyEfficientNetB5.py        | My EfficientNetB5 implementation     |
| **backbone**   | EfficientNet   | MyEfficientNetB6.py        | My EfficientNetB6 implementation     |
| **backbone**   | EfficientNet   | MyEfficientNetB7.py        | My EfficientNetB7 implementation     |
| **backbone**   | EfficientNetV2 | MyEfficientNetV2L.py       | My EfficientNetV2L implementation    |
| **backbone**   | EfficientNetV2 | MyEfficientNetV2M.py       | My EfficientNetV2M implementation    |
| **backbone**   | EfficientNetV2 | MyEfficientNetV2S.py       | My EfficientNetV2S implementation    |
| **backbone**   | ResNet         | ResNet18.py                | My ResNet18 implementation           |
| **backbone**   | ResNet         | ResNet34.py                | My ResNet34 implementation           |
| **backbone**   | ResNet         | ResNet50.py                | My ResNet50 implementation           |
| **backbone**   | ResNet         | ResNet101.py               | My ResNet101 implementation          |
| **backbone**   | ResNet         | ResNet152.py               | My ResNet152 implementation          |
| **backbone**   | ResNeXt        | MyResNeXt50_32x4d.py       | My ResNeXt50_32x4d implementation    |
| **backbone**   | ResNeXt        | MyResNeXt101_32x8d.py      | My ResNeXt101_32x8d implementation   |
| **backbone**   | ResNeXt        | MyResNeXt101_64x4d.py      | My ResNeXt101_64x4d implementation   |
| **backbone**   | Swin           | MySwinB.py                 | My SwinB implementation              |
| **backbone**   | Swin           | MySwinS.py                 | My SwinS implementation              |
| **backbone**   | Swin           | MySwinT.py                 | My SwinT implementation              |
| **backbone**   | -              | MyDenseNet_models.py       | Select DenseNet models               |
| **backbone**   | -              | MyEfficientNet_models.py   | Select EfficientNet models           |
| **backbone**   | -              | MyEfficientNetV2_models.py | Select EfficientNetV2 models         |
| **backbone**   | -              | MyResNet_models.py         | Select ResNet models                 |
| **backbone**   | -              | MyResNeXt_models.py        | Select ResNeXt models                |
| **backbone**   | -              | MySwin_models.py           | Select Swin models                   |
| **gravitynet** | -              | ClassificationSubNet.py    | Classification SubNet implementation |
| **gravitynet** | -              | GravityNet.py              | GravityNet implementation            |
| **gravitynet** | -              | RegressionSubNet.py        | Regression SubNet implementation     |
| **utility**    | -              | load_model.py              | load best-model                      |
| **utility**    | -              | my_torchsummary.py         | save model summary                   |
| **utility**    | -              | save_model.py              | save best-model                      |