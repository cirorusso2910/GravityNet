# NET.INITIALIZATION
**net.initialization** package

## STRUCTURE

    initialization
    |
    | - dict
    |   | metrics.py
    |   | plot_title.py
    |   | plot_title_complete.py
    |
    | - folders
    |   | dataset_folders.py
    |   | default_folders.py
    |   | experiment_complete_folders.py
    |   | experiment_folders.py
    |
    | - header
    |   | annotations.py
    |   | coords.py
    |   | detections.py
    |   | metrics.py
    |   | statistics.py
    |
    | - ID
    |   | experimentID.py
    |   | experimentID_complete.py
    |   | experimentID_fold.py
    |
    | - path
    |   | detections_fold_path.py
    |   | experiment_complete_result_path.py
    |   | experiment_result_path.py
    |   | test_NMS_result_path.py 
    |
    | - utility
    |   | create_folder.py
    |   | create_folder_and_subfolder.py
    |   | get_parametersID.py
    |   | parametersID.py
    |
    | init.py
    | init_complete.py

## DOCUMENTATION

| FOLDER      | FUNCTION                           | DESCRIPTION                                                                  |
|-------------|------------------------------------|------------------------------------------------------------------------------|
| **dict**    | metrics.py                         | Get metrics dictionary according to type                                     |
| **dict**    | plot_title.py                      | Define plot title experiment results                                         |
| **dict**    | plot_title_complete.py             | Define plot title experiment complete results                                |
| **folders** | dataset_folders.py                 | Example of dataset folders dictionary                                        |
| **folders** | default_folders.py                 | Default folders dictionary                                                   |
| **folders** | experiment_complete_folders.py     | Experiment complete folders dictionary                                       |
| **folders** | experiment_folders.py              | Experiment folders dictionary                                                |
| **header**  | annotations.py                     | Get annotation header                                                        |
| **header**  | coords.py                          | Get coords header                                                            |
| **header**  | detections.py                      | Get detections header                                                        |
| **header**  | metrics.py                         | Get metrics header according to type                                         |
| **header**  | statistics.py                      | Get statistics header                                                        |
| **ID**      | experimentID.py                    | Concatenate experiment-ID according to type                                  |
| **ID**      | experimentID_complete.py           | Concatenate experiment-ID complete                                           |
| **ID**      | experimentID_fold.py               | Get experiment-1-fold-ID and experiment-2-fold-ID                            |
| **path**    | detections_fold_path.py            | Concatenate detections 1-fold and 2-fold path                                |
| **path**    | experiment_complete_result_path.py | Concatenate experiment complete result path                                  |
| **path**    | experiment_result_path.py          | Concatenate experiment result path                                           |
| **path**    | test_NMS_result_path.py            | Concatenate Test-NMS result path                                             |
| **utility** | create_folder.py                   | Create folder                                                                |
| **utility** | create_folder_and_subfolder.py     | Create folder (main-path) and subfolder (subfolder-path-dict) of main-folder |
| **utility** | get_parametersID.py                | Get parameters-ID from experiment-ID                                         |
| **utility** | parametersID.py                    | Get parameters ID                                                            |
| -           | init.py                            | Initialization of experiment results folder based on execution mode          |
| -           | init_complete.py                   | Initialization of experiment complete results folder                         |
