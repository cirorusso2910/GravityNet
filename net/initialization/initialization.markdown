# NET.INITIALIZATION
**net.initialization** package

## STRUCTURE

    initialization
    |
    | - dict
    |   | metrics.py
    |   | plot_title.py
    |
    | - folders
    |   | dataset_folders.py
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
    |
    | - path
    |   | experiment_result_path.py
    |
    | - utility
    |   | create_folder.py
    |   | create_folder_and_subfolder.py
    |   | parametersID.py
    |
    | init.py

## DOCUMENTATION

| FOLDER      | FUNCTION                           | DESCRIPTION                                                                  |
|-------------|------------------------------------|------------------------------------------------------------------------------|
| **dict**    | metrics.py                         | Get metrics dictionary according to type                                     |
| **dict**    | plot_title.py                      | Define plot title experiment results                                         |
| **folders** | dataset_folders.py                 | Dataset folders dictionary                                                   |
| **folders** | experiment_folders.py              | Experiment folders dictionary                                                |
| **header**  | annotations.py                     | Get annotation header                                                        |
| **header**  | coords.py                          | Get coords header                                                            |
| **header**  | detections.py                      | Get detections header                                                        |
| **header**  | metrics.py                         | Get metrics header according to type                                         |
| **header**  | statistics.py                      | Get statistics header                                                        |
| **ID**      | experimentID.py                    | Concatenate experiment-ID according to type                                  |
| **path**    | experiment_result_path.py          | Concatenate experiment result path                                           |
| **utility** | create_folder.py                   | Create folder                                                                |
| **utility** | create_folder_and_subfolder.py     | Create folder (main-path) and subfolder (subfolder-path-dict) of main-folder |
| **utility** | parametersID.py                    | Get parameters ID                                                            |
| -           | init.py                            | Initialization of experiment results folder based on execution mode          |