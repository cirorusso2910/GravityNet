# NET.DEBUG
**net.debug** package

## STRUCTURE

    debug
    |
    | debug_anchors.py
    | debug_detections.py
    | debug_execution.py
    | debug_hooking.py
    | debug_network_summary.py

## DOCUMENTATION

| FUNCTION                 | DESCRIPTION                                                                                                                     |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| debug_anchors.py         | save: initial gravity points configuration (image and csv),  gravity points configuration (image and csv), feature grid (image) |
| debug_detections.py      | save: single image detections (in ./debug folder)                                                                               |
| debug_execution.py       | stop execution                                                                                                                  |
| debug_hooking.py         | save: image with gravity points configuration and colors the gravity points hooked with 'similar' color                         |
| debug_network_summary.py | show and save: summary network                                                                                                  |
