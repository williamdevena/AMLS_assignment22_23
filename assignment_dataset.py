"""
This module contains the class Dataset that represents the possible combinations of dataset and label:
- 'cartoon' dataset with 'eye_color' or 'face_shape' labels
- 'celeba' dataset with 'smiling' or 'gender' labels
"""

class AssignmentDataset():
    
    possible_combinations = {
        # "cartoon": (['eye_color', 'face_shape'], (500, 500)),
        # #"cartoon": (['eye_color', 'face_shape'], (250, 250)),
        "celeba": (['gender', 'smiling'], (178, 218)),
        # "cropped_eye_cartoon": (['eye_color'], (25, 29)),
        "cropped_mouth_celeba": (['gender', 'smiling'], (58, 33)),
        "cropped_eyes_celeba": (['gender', 'smiling'], (98, 38)),
        "dyn_cropped_mouth_celeba": (['gender', 'smiling'], (48, 25)),
        "dyn_cropped_eyes_celeba": (['gender', 'smiling'], (65, 25)),
        # "dominant_color": (['eye_color'], None)
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
        "eye_color": [],
        "face_shape": [],
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
        