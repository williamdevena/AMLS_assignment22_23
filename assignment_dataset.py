"""
This module contains the class Dataset that represents the possible combinations of dataset and label:
- 'cartoon' dataset with 'eye_color' or 'face_shape' labels
- 'celeba' dataset with 'smiling' or 'gender' labels
"""

from src import costants

class AssignmentDataset():
    
    raw_possible_combinations = {
        "cartoon": (['eye_color', 'face_shape'], (500, 500)),
        #"cartoon": (['eye_color', 'face_shape'], (250, 250)),
        "celeba": (['gender', 'smiling'], (178, 218)),
    }
    
    possible_combinations = {
        "cartoon": (['eye_color', 'face_shape'], (500, 500)),
        # "cartoon": ([
        #             #'eye_color', 
        #              'face_shape'
        #             ],
        #            (250, 250)),
        "celeba": (['gender', 'smiling'], (178, 218)),
        "cropped_eye_cartoon": (['eye_color'], (25, 29)),
        "dyn_cropped_eye_cartoon": (['eye_color'], (30, 30)),
        "dyn_cropped_face_cartoon": (['face_shape'], (125, 140)),
        
        "cropped_mouth_celeba": ([
            'gender', 
            'smiling'], (58, 33)),
        "cropped_eyes_celeba": (['gender', 'smiling'], (98, 38)),
        "dyn_cropped_mouth_celeba": (['gender', 'smiling'], (48, 25)),
        "dyn_cropped_eyes_celeba": (['gender', 'smiling'], (65, 25)),
        "dyn_cropped_face_celeba": (['gender', 'smiling'], (125, 140)),
    }
    
    
    paths = {
        "cartoon": {
            "train_images": costants.PATH_CARTOON_TRAIN_IMG,
            "train_labels": costants.PATH_CARTOON_TRAIN_LABELS,
            "test_images": costants.PATH_CARTOON_TEST_IMG,
            "test_labels": costants.PATH_CARTOON_TEST_LABELS,
        },
        "celeba": {
            "train_images": costants.PATH_CELEBA_TRAIN_IMG,
            "train_labels": costants.PATH_CELEBA_TRAIN_LABELS,
            "test_images": costants.PATH_CELEBA_TEST_IMG,
            "test_labels": costants.PATH_CELEBA_TEST_LABELS,
        },
        "cropped_eye_cartoon": {
            "train_images": costants.PATH_CARTOON_TRAIN_CROPPED_EYE_IMG,
            "train_labels":  costants.PATH_CARTOON_TRAIN_LABELS,
            "test_images": costants.PATH_CARTOON_TEST_CROPPED_EYE_IMG,
            "test_labels" : costants.PATH_CARTOON_TEST_LABELS
        },
        "dyn_cropped_eye_cartoon": {
            "train_images": costants.PATH_CARTOON_TRAIN_DYN_CROPPED_EYE_IMG,
            "train_labels": costants.PATH_CARTOON_TRAIN_LABELS,
            "test_images": costants.PATH_CARTOON_TEST_DYN_CROPPED_EYE_IMG,
            "test_labels": costants.PATH_CARTOON_TEST_LABELS,
        },
        "cropped_mouth_celeba": {
            "train_images": costants.PATH_CELEBA_TRAIN_CROPPED_MOUTH_IMG,
            "train_labels": costants.PATH_CELEBA_TRAIN_LABELS,
            "test_images": costants.PATH_CELEBA_TEST_CROPPED_MOUTH_IMG,
            "test_labels": costants.PATH_CELEBA_TEST_LABELS,
        },
        "cropped_eyes_celeba": {
            "train_images": costants.PATH_CELEBA_TRAIN_CROPPED_EYES_IMG,
            "train_labels": costants.PATH_CELEBA_TRAIN_LABELS,
            "test_images": costants.PATH_CELEBA_TEST_CROPPED_EYES_IMG,
            "test_labels": costants.PATH_CELEBA_TEST_LABELS,
        },   
        "dyn_cropped_eyes_celeba": {
            "train_images": costants.PATH_CELEBA_TRAIN_DYN_CROPPED_EYES_IMG,
            "train_labels": costants.PATH_CELEBA_TRAIN_LABELS,
            "test_images": costants.PATH_CELEBA_TEST_DYN_CROPPED_EYES_IMG,
            "test_labels": costants.PATH_CELEBA_TEST_LABELS,
        },
        "dyn_cropped_mouth_celeba": {
            "train_images": costants.PATH_CELEBA_TRAIN_DYN_CROPPED_MOUTH_IMG,
            "train_labels": costants.PATH_CELEBA_TRAIN_LABELS,
            "test_images": costants.PATH_CELEBA_TEST_DYN_CROPPED_MOUTH_IMG,
            "test_labels": costants.PATH_CELEBA_TEST_LABELS,
        }
    }
    
    
    best_knn_dataset_label = {
        "celeba": {
            "gender": 40,
            "smiling": 20,
        },
        "cropped_mouth_celeba": {
            "gender": 30,
            "smiling": 30,
        },
        "cropped_eyes_celeba": {
            "gender": 30,
            "smiling": 20,
        },
        "cropped_eye_cartoon": {
            "eye_color": 20,
            # "face_shape": 
        },
        "dyn_cropped_eye_cartoon": {
            "eye_color": 20,
            # "face_shape": 
        },
        "dyn_cropped_eyes_celeba": {
            "gender": 30,
            "smiling": 20,
        },
        "dyn_cropped_eye_cartoon": {
            "eye_color": 20,
            # "face_shape": 
        },
       # "cartoon": {},
       # "dominant_color": {}
        
    }
    
    labels_values = {
        "eye_color": ["0", "1", "2", "3", "4"],
        "face_shape": ["0", "1", "2", "3", "4"],
        "gender": ["female", "male"],
        "smiling": ["non smiling", "smiling"]
    }
    
    def __init__(self, name, label) -> None:
        if name in AssignmentDataset.possible_combinations:
            if label in AssignmentDataset.possible_combinations[name][0]:
                self.name = name 
                self.label = label
            else:
                raise Exception(
                    f"The label of the dataset {name} has a non possible value ({label})."
                )
        else:
            raise Exception(
                f"The dataset name has a non possible value ({name})."
            )
            
        self.image_dimensions = AssignmentDataset.possible_combinations[name][1]
        