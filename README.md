# Gender and Smiling detection (AMLS_assignment22_23)
Assignment project for the course Applied Machine Learning Systems I. Read the file "ELEC134(22-23)-new.pdf" for more details on the assignment.
Read the file "Report.pdf" and "Additional_material.pdf" for a detailed description of the approaches used.

## Requirements

- See requirements.txt (created with command conda liste -e)

## Instructions for running the code

- Run main.py

## Brief explanation of main.py

- Lines 67-108 create the "static cropped" dataset mentioned in the report

- 110-230 create the "dynamic cropping" datasets

- 230-244 create the landmark features datasets

- 246-359 run SVM and KNN on the landmark features, for the gender detection, smile detection and face shape recognition tasks. The results are both printed on the screen and written in the file log/assignment.log

- Line 373 (knn_for_every_dataset) runs KNN on all the cropped and non-cropped (original) dataset, for all four tasks. The results are written in the folder plots_and_logs/knn

- Line 373 (knn_for_every_dataset) runs KNN on all the cropped and non-cropped (original) dataset, for all four tasks. The results are written in the folder plots_and_logs/svm

- 391-465 Load the weights of the two trained models from the folder weights_cnn and runs testing for the gender detection and face shape recognition tasks.
