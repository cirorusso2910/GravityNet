# EXPERIMENTS-STRUCTURE

Each **experiment** has the following structure:

    detections
        -> detections folder
    explainability [explainability mode]
        -> explainability folder
    log
        -> log folder
    metrics-train
        -> metrics-train folder
    metrics-test
        -> metrics-test folder
    models
        -> models folder
    output
        -> output folder
    plots-test
        -> plots-test folder
    plots-train
        -> plots-train folder
    plots-validation
        -> plots-validation folder


- The **output** folder has the following structure:
    
        output
           | - output-gravity-test
           | - output-gravity-validation
           | - output-test
 
- The **plots-test** folder has the following structure:

        plots-test
            | - coords

- The **plots-validation** folder has the following structure:

        plots-validation
            | - coords
            |    | - coords-FROC-validation
            |    | - coords-PR-validation
            |    | - coords-ROC-validation
            |
            | - FROC-validation
            | - PR-validation
            | - ROC-validation