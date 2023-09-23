# NET.ANCHORS
**net.anchors** package

## STRUCTURE

    anchors
    |
    | - draw
    |   | draw_gravity_points.py
    |
    | - initial_config
    |   | initial_dice_config.py
    |   | initial_grid_config.py
    |
    | - utility
    |   | check_image_shape.py
    |   | get_feature_grid.py
    |   | get_num_gravity_points.py
    |   | shift.py
    |
    | gravity_points_config.py
    | gravity_points_prediction.py

## DOCUMENTATION

| FOLDER             | FUNCTION                     | DESCRIPTION                                                                                                                                                                          |
|--------------------|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **draw**           | draw_gravity_points.py       | Draw gravity points configuration on image and save                                                                                                                                  |
| **initial_config** | initial_dice_config.py       | Generate initial gravity points with dice configuration in a reference window (feature map)                                                                                          |
| **initial_config** | initial_grid_config.py       | Generate initial gravity points with grid configuration in a reference window (feature map)                                                                                          |
| **utility**        | check_image_shape.py         | Check if image shape dimension is multiple of 32. Otherwise, the gravity points are not placed within the image                                                                      |
| **utility**        | get_feature_grid.py          | Get grid shifts on image in x and y                                                                                                                                                  |
| **utility**        | get_num_gravity_points.py    | Get num of gravity points per feature grid and per image                                                                                                                             |
| **utility**        | shift.py                     | Shift initial configuration over the whole image: build a shift meshgrid and shift the initial configuration in a reference window (feature map) over the whole image                |
| -                  | gravity_points_config.py     | Generate gravity points configuration: initial configuration is generated which is then shift on the feature map so that each pixel of the feature map has the initial configuration |
| -                  | gravity_points_prediction.py | Compute gravity points prediction                                                                                                                                                    |
