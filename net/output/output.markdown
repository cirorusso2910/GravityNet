# NET.PLOT
**net.plot** package

## STRUCTURE

    plot
    |
    | - utility
    |   | select_image_channel.py
    |   | select_output_gravity_filename.py
    |
    | output.py
    | output_FPS.py
    | output_gravity.py
    | output_score_gravity.py

## DOCUMENTATION

| FOLDER       | FUNCTION                          | DESCRIPTION                                                                |
|--------------|-----------------------------------|----------------------------------------------------------------------------|
| **utility**  | select_image_channel.py           | Select image channel                                                       |
| **utility**  | select_output_gravity_filename.py | Select filename to save output gravity                                     |
| -            | output.py                         | Save detections output results                                             |
| -            | output_FPS.py                     | Save detections output results at specific false positive per scan (FPS)   |
| -            | output_gravity.py                 | Save detections output gravity results for a single image detections       |
| -            | output_score_gravity.py           | Save detections output score gravity results for a single image detections |