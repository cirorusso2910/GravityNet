# NET.DETECTIONS
**net.detections** package

## STRUCTURE

    detections
    |
    | - utility
    |   | check_index.py
    |   | conversion_item.py
    |   | get_single_detection.py
    |   | init_detections.py
    |
    | detections_test_distance.py
    | detections_validation_distance.py

## DOCUMENTATION

| FOLDER      | FUNCTION                            | DESCRIPTION                                                                                                                    |
|-------------|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **utility** | check_index.py                      | Check index and delete elements in index_positive and index_TP: to avoid one prediction hooking into two different annotations |
| **utility** | conversion_item.py                  | Converts single detections item with distance evaluation                                                                       |
| **utility** | get_single_detection.py             | Get single detection for specific index                                                                                        |
| **utility** | init_detections.py                  | initialize detections distance and radius                                                                                      |
|             | detections_test_distance.py         | Compute detections in test with DISTANCE metrics and save in detections.csv                                                    |
|             | detections_validation_distance.py   | Compute detections in validation with DISTANCE metrics and save in detections.csv                                              |
