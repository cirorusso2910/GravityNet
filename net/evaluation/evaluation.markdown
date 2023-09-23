# NET.EVALUATION
**net.evaluation** package

## STRUCTURE

    evaluation
    |
    | - utility
    |   | distance_eval_rescale.py
    |   | radius_eval.py
    |   | select_FP_list_path.py
    |   | select_TotalNumOfImages.py
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

| FOLDER      | FUNCTION                   | DESCRIPTION                                                                                      |
|-------------|----------------------------|--------------------------------------------------------------------------------------------------|
| **utility** | distance_eval_rescale.py   | Rescale distance for evaluation according to rescale factor                                      |
| **utility** | radius_eval.py             | Get radius factor from eval parameter                                                            |
| **utility** | select_FP_list_path.py     | Select list of images which calculate False Positive (FP)                                        |
| **utility** | select_TotalNumOfImages.py | Select TotalNumOfImages for FROC computation based on where False Positive (FP) where calculated |
|             | AUC.py                     | Compute Area Under the Curve (AUC)                                                               |
|             | AUFROC.py                  | Compute Area Under the FROC Curve (AUFROC) in range [0, FPS-upper-bound]                         |
|             | current_learning_rate.py   | Get current learning rate according to scheduler and optimizer type                              |
|             | FROC.py                    | Compute FROC Curve                                                                               |
|             | FROC_FPS_index.py          | Get FROC sensitivity at specific FPS and score threshold of specific FPS                         |
|             | ROC.py                     | Compute ROC Curve                                                                                |
|             | sensitivity.py             | Compute sensitivity at work point and sensitivity max                                            |
|             | sensitivity_images.py      | Compute sensitivity per image                                                                    |