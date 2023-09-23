# NET.DETECTIONS
**net.detections** package

## STRUCTURE

    detections
    |
    | - utility
    |   | check_index.py
    |   | conversion_item.py
    |   | detections_concatenation.py
    |   | get_single_detection.py
    |   | init_detections.py
    |   | read_detections.py
    |
    | AUC.py
    | AUFROC.py
    | current_learning_rate.py
    | FROC.py
    | FROC_FPS_index.py
    | ROC.py
    | sensitivity.py
    | sensitivity_images.py

## DOCUMENTATION

| FOLDER      | FUNCTION                            | DESCRIPTION                                                                                                                    |
|-------------|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **utility** | check_index.py                      | Check index and delete elements in index_positive and index_TP: to avoid one prediction hooking into two different annotations |
| **utility** | conversion_item.py                  | Converts single detections item with distance evaluation                                                                       |
| **utility** | detections_concatenation.py         | Save detections 1-fold and 2-fold concatenation                                                                                |
| **utility** | get_single_detection.py             | Get single detection for specific index                                                                                        |
| **utility** | init_detections.py                  | initialize detections distance and radius                                                                                      |
| **utility** | read_detections.py                  | Read detections file (.csv) according to type                                                                                  |
|             | detections_test_1_image_distance.py | Compute detections in test 1-image with DISTANCE metrics and save in detections.csv                                            |
|             | detections_test_1_image_radius.py   | Compute detections in test 1-image with RADIUS metrics and save in detections.csv                                              |
|             | detections_test_distance.py         | Compute detections in test with DISTANCE metrics and save in detections.csv                                                    |
|             | detections_test_NMS_distance.py     | Compute detections in test NMS with DISTANCE metrics and save in detections.csv                                                |
|             | detections_test_NMS_radius.py       | Compute detections in test NMS with RADIUS metrics and save in detections.csv                                                  |
|             | detections_test_radius.py           | Compute detections in test with RADIUS metrics and save in detections.csv                                                      |
|             | detections_validation_distance.py   | Compute detections in validation with DISTANCE metrics and save in detections.csv                                              |
|             | detections_validation_radius.py     | Compute detections in validation with RADIUS metrics and save in detections.csv                                                |