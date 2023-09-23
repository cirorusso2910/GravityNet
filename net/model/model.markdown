# NET.MODEL
**net.model** package

## STRUCTURE

    model
    |
    | - backbone
    |   | - ResNet
    |   |   | MyResNet18.py
    |   |   | MyResNet34.py
    |   |   | MyResNet50.py
    |   |   | MyResNet101.py
    |   |   | MyResNet152.py
    |   
    | - gravitynet
    |   | ClassificationSubNet.py
    |   | GravityNet.py
    |   | RegressionSubNet.py
    |
    | - utility
    |   | load_model.py
    |   | my_torchsummary.py
    |   | save_model.py
    |
    | MyResNet_models.py

## DOCUMENTATION

| FOLDER       | FUNCTION                          | DESCRIPTION                                                          |
|--------------|-----------------------------------|----------------------------------------------------------------------|
| **utility**  | select_image_channel.py           | Select image channel                                                 |
| **utility**  | select_output_gravity_filename.py | Select filename to save output gravity                               |
| -            | output.py                         | Save detections output results                                       |
| -            | output_gravity.py                 | Save detections output gravity results for a single image detections |
