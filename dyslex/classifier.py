import io
import json
import numpy as np
import pickle
from PIL import Image
import os
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import feature_extractor as feat_ext
import models as models
from utils import load_sklearn_classifier

# constants
fill_null_values_by_zero = True
summarize_characteristics_by_mean = True
normalize_values = False
generate_fixation_image_for_each_trialid = False

def extract_subject_feature_for_task(task_type_id, task_feature_def_dict, subject_id_col_name, aoi_id_col_name, feature_file_path_metrics):

    # Feature extraction -- pre-defined features
    subject_id, subject_data_dict = feat_ext.load_and_transform_subject_characteristics(task_type_id, task_feature_def_dict, fill_null_values_by_zero, subject_id_col_name, aoi_id_col_name, summarize_characteristics_by_mean, feature_file_path_metrics)
    df = feat_ext.create_subject_characteristics_profile(subject_id, subject_data_dict)
    X = df.drop('subject_id', axis=1).astype(np.float64)
    #print(f'Feature count: {len(X.columns)}')
    return X

def extract_subject_fixation_image_for_task(task_type_id, fixation_image_characteristics_names, degrees_visual_angle_pixels, fixation_image_visual_params, subject_id_col_name, feature_file_path_fixations):
    # Feature extraction -- fixation images
    # Task-specific variables for the construction of fixation image
    x_min, x_max, y_min, y_max, d_max = fixation_image_visual_params[task_type_id]
    fixation_duration_color_norm = Normalize(0, d_max)
    # Subject data
    subject_id, figs_dict, df_fixations_all = feat_ext.generate_subject_fixation_images(generate_fixation_image_for_each_trialid, fixation_image_characteristics_names, fill_null_values_by_zero, subject_id_col_name, degrees_visual_angle_pixels, fixation_duration_color_norm, x_min, x_max, y_min, y_max, feature_file_path_fixations)
    # Expect a single fixation image per experiment (i.e., generate_fixation_image_for_each_trialid = False)
    fig = figs_dict[-1]
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', facecolor='white', transparent=False, pad_inches=0) # Create a PIL (Pillow) Image object from the byte array
    plt.close(fig)
    fig = Image.open(img_buf).convert('RGB')
    
    ## Save the fixation image to a file
    # fig.save(f'fixation_image_{task_type_id}.png')
    
    # Fixation image transformation
    fixation_image_transforms = feat_ext.FixationImageTransform()
    fig = fixation_image_transforms(fig)
    fig = fig.unsqueeze(0)
    return fig

def classify_subject_for_task(task_models_def, task_id, feature_predefined, feature_fixation_image, print_results=True, str_to_print_with_each_result=""):
    results: dict = {}

    # Classification
    for task_model_def in task_models_def[task_id]:
        type_id = task_model_def['type_id']

        # Data scaling
        scaler = task_model_def['scaler']
        X_scaled = None
        if scaler is None:
            X_scaled = feature_predefined
        else:
            X_scaled = torch.from_numpy(scaler.transform(feature_predefined.values))
            X_scaled = X_scaled.float() # Convert X to float64

        probs = None
        preds = None

        # Case 1: kNN
        if type_id == 'kNN':
            probs = torch.from_numpy(task_model_def['loaded_model'].predict_proba(X_scaled))
            probs, preds = torch.max(probs, 1)

        # Case 2: MLP
        if type_id == 'MLP':
            with torch.no_grad():
                outputs = task_model_def['loaded_model'](X_scaled)
                outputs = outputs.softmax(dim=1)
                probs, preds = torch.max(outputs, 1)

        # Case 3: CNN
        if type_id.startswith('CNN-'):
            with torch.no_grad():
                outputs = task_model_def['loaded_model'](feature_fixation_image)
                outputs = outputs.softmax(dim=1)
                probs, preds = torch.max(outputs, 1)

        # Save the results
        results[task_model_def['desc_short']] = {
            'preds': preds.squeeze().tolist(),
            'probs': probs.squeeze().tolist()
        }

        # Prints the classification results
        if print_results:
            print(f'  {task_model_def["desc_short"]}: {preds.squeeze()} ({probs.squeeze():.3f}); {str_to_print_with_each_result}')

    return results

# main
def main():

    # Read configuration properties from a json file
    with open('properties.json') as property_file:
        properties = json.loads(property_file.read())

    # Path to directory with metainformation about the dataset and features for individual tasks
    meta_dir = properties['meta_dir']
    # Path to directory with trained models
    models_dir = properties['models_dir']

    # Data properties
    aoi_id_col_name = properties['aoi_id_col_name']
    subject_id_col_name = properties['subject_id_col_name']
    degrees_visual_angle_pixels = properties['degrees_visual_angle_pixels']

    # Fixation image parameters
    fixation_image_characteristics_names = [
        properties['fixation_image_fix_x_col_name'],
        properties['fixation_image_fix_y_col_name'],
        properties['fixation_image_disp_x_col_name'],
        properties['fixation_image_disp_y_col_name'],
        properties['fixation_image_duration_ms_col_name']
        ]
    fixation_image_visual_params = eval(properties['fixation_image_visual_params'])

    # Definitions of tasks and trained models
    tasks_def = eval(properties['tasks'])
    task_feature_def_dict = feat_ext.load_task_feature_definition_dict(tasks_def, meta_dir)
    task_models_def = eval(properties['task_models'])

    # Load the trained models
    for task_id in task_models_def.keys():
        for task_model_def in task_models_def[task_id]:
            type_id = task_model_def['type_id']
            file_name = task_model_def['file_name']
            desc_short = task_model_def['desc_short']
            print(f'Loading model: {desc_short} ({file_name})')

            # Load the data scaler from file
            if 'scaler_file_name' not in task_model_def.keys():
                task_model_def['scaler'] = None
            else:
                task_model_def['scaler'] = pickle.load(open(os.path.join(models_dir, task_model_def['scaler_file_name']), 'rb'))
                print(f"  Loading scalar: {task_model_def['scaler_file_name']}")

            # Load the classifier
            classifier = None
            # Case 1: kNN
            if type_id == 'kNN':
                classifier = load_sklearn_classifier(models_dir, file_name)
            # Case 2: MLP
            elif type_id == 'MLP':
                params = task_model_def['params'].split(';')
                classifier = models.MLP(int(params[0]), int(params[1]), int(params[2]), int(params[3]), float(params[4]))
                classifier.load_state_dict(torch.load(os.path.join(models_dir, file_name), map_location=torch.device('cpu')))
                classifier.eval()
            # Case 3: ResNet18
            elif type_id == 'CNN-RN18':
                classifier = models.binary_resnet18()
                classifier.load_state_dict(torch.load(os.path.join(models_dir, file_name), map_location=torch.device('cpu')))
                classifier.eval()
            elif type_id == 'CNN-RN50':
                classifier = models.binary_resnet50()
                classifier.load_state_dict(torch.load(os.path.join(models_dir, file_name), map_location=torch.device('cpu')))
                classifier.eval()
            else:
                classifier = None

            task_model_def['loaded_model'] = classifier
            if classifier is not None:
                print(f'Classifier successfully loaded.')
            else:
                print(f'Classifier not loaded.')

    # Batch classification
    input_feature_file_path = f"y:/datasets/dyslex/experiment-final/v3.0/original-all/"
    #subject_ids_to_classify = '[1003 1009 1019 1021 1033 1038 1040 1058 1065 1090 1109 1113 1115 1145 1160 1166 1169 1174 1186 1187 1209 1235 1254 1255 1257 1258 1263 1274 1284 1312 1314 1318 1322 1345 1349 1350 1380 1398 1405 1417 1421 1459 1582 1591 1626 1693 1729 1744 1760 1859 1869 1879 1903 1929 1993 1996]' # all folds
    #subject_ids_to_classify = '[1038 1145 1160 1166 1187 1209 1235 1255 1318 1405 1693 1869 1993 1996]' # fold1
    #subject_ids_to_classify = '[1016 1073 1075 1082 1095 1134 1189 1271 1300 1377 1476 1571 1858 1913]' # fold2
    #subject_ids_to_classify = '[1009 1021 1040 1065 1090 1169 1174 1284 1322 1380 1591 1626 1879 1929]' # fold3
    #subject_ids_to_classify = '[1003 1019 1033 1113 1115 1186 1258 1274 1312 1314 1349 1417 1459 1859]' # fold4
    #subject_ids_to_classify = '[1058 1109 1254 1257 1263 1345 1350 1398 1421 1582 1729 1744 1760 1903]' # fold5
    #subject_ids_to_classify = '[1496]' # class0 (subject NOT present in training/validation data)
    subject_ids_to_classify = '[1686]' # class0 (subject NOT present in training/validation data)
    #subject_ids_to_classify = '[1571]' # class1 (subject present in training/validation data)
    #subject_ids_to_classify = '[1858]' # class1 (subject present in training/validation data)

    # Convert the string to the string array
    subject_ids_to_classify = np.fromstring(subject_ids_to_classify[1:-1], dtype=int, sep=' ')
    for task_def in tasks_def:
        task_id = task_def['id']
        task_type_id = task_def['type_id']
        print(f'Classification task: {task_id}')

        for subject_id_to_classify in subject_ids_to_classify:
            feature_file_path_metrics = f"{input_feature_file_path}Subject_{subject_id_to_classify}_{task_id}_metrics.csv"
            feature_file_path_fixations = f"{input_feature_file_path}Subject_{subject_id_to_classify}_{task_id}_fixations.csv"

            # Extract features
            feature_predefined = extract_subject_feature_for_task(task_type_id, task_feature_def_dict, subject_id_col_name, aoi_id_col_name, feature_file_path_metrics)
            feature_fixation_image = extract_subject_fixation_image_for_task(task_type_id, fixation_image_characteristics_names, degrees_visual_angle_pixels, fixation_image_visual_params, subject_id_col_name, feature_file_path_fixations)
            classify_subject_for_task(task_models_def, task_id, feature_predefined, feature_fixation_image, print_results=True, str_to_print_with_each_result=f'subject_id_to_classify: {subject_id_to_classify}')

if __name__ == "__main__":
    main()
