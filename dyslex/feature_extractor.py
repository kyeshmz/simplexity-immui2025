from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from torchvision import transforms

def load_task_feature_definition_dict(tasks_def, meta_dir):

    task_feature_def_dict = {}
    for task_type_id in [task['type_id'] for task in tasks_def]:

        # Definition files for the task
        definition_characteristics_global_file = os.path.join(meta_dir, 'definition-' + task_type_id + '-characteristics-global.txt')
        definition_characteristics_file = os.path.join(meta_dir, 'definition-' + task_type_id + '-characteristics.txt')
        definition_AOIs_file = os.path.join(meta_dir, 'definition-' + task_type_id + '-AOIs.txt')
        definition_AOIs_characteristics_file = os.path.join(meta_dir, 'definition-' + task_type_id + '-AOIs-characteristics.txt')

        # Read the names of characteristics-global if the file exists
        if not os.path.exists(definition_characteristics_global_file):
            characteristics_global = None
        else:
            #characteristics_global = list(np.loadtxt(definition_characteristics_global_file, dtype=str, comments='#', delimiter='\n'))
            characteristics_global = list(np.loadtxt(definition_characteristics_global_file, dtype=str, comments='#'))
            #with open(definition_characteristics_global_file) as f:
            #    characteristics_global = [line.rstrip() for line in f if not line.startswith('#') and line.rstrip() != '']
            print(f'* characteristics-global: {len(characteristics_global)}')

        # Read the names of characteristics if the file exists
        if not os.path.exists(definition_characteristics_file):
            characteristics = None
            characteristics_expected_line_count = None
        else:
            #characteristics = list(np.loadtxt(definition_characteristics_file, dtype=str, comments='#', delimiter='\n'))
            characteristics = list(np.loadtxt(definition_characteristics_file, dtype=str, comments='#'))
            # Get the number of expected lines with characteristics
            characteristics_expected_line_count = int(characteristics.pop(0))
            print(f'* characteristics: {len(characteristics)}, expected line count: {characteristics_expected_line_count}')

        # Read the names of AOIs and AOI characteristics if the file exists
        if (not os.path.exists(definition_AOIs_file)) or (not os.path.exists(definition_AOIs_characteristics_file)):
            aoi_names = None
            aoi_characteristics = None
        else:
            #aoi_names = list(np.loadtxt(definition_AOIs_file, dtype=str, comments='#', delimiter='\n'))
            aoi_names = list(np.loadtxt(definition_AOIs_file, dtype=str, comments='#'))
            #aoi_characteristics = list(np.loadtxt(definition_AOIs_characteristics_file, dtype=str, comments='#', delimiter='\n'))
            aoi_characteristics = list(np.loadtxt(definition_AOIs_characteristics_file, dtype=str, comments='#'))
            print(f'* AOI characteristics: {len(aoi_characteristics)}, AOIs: {len(aoi_names)}')
        
        task_feature_def_dict[task_type_id] = {
            'characteristics_global': characteristics_global,
            'characteristics': characteristics,
            'characteristics_expected_line_count': characteristics_expected_line_count,
            'aoi_names': aoi_names,
            'aoi_characteristics': aoi_characteristics
        }
    
    return task_feature_def_dict

def load_and_transform_subject_characteristics(task_type_id, task_feature_def_dict, fill_null_values_by_zero, subject_id_col_name, aoi_id_col_name, summarize_characteristics_by_mean, subject_characteristics_file_path):

    # Read the subject data from a csv file
    df_original = pd.read_csv(subject_characteristics_file_path, skiprows=0, sep=',')

    # Fill the missing (nan) values with 0
    if fill_null_values_by_zero:
        df_original = df_original.fillna(0.0)

    # Get the subject id from the first line
    subject_id = str(df_original[subject_id_col_name].iloc[0])

    characteristics_dict = {'characteristics_global': None, 'characteristics': None, 'AOIs_characteristics': None}

    # Characteristics-global
    characteristics_global = task_feature_def_dict[task_type_id]['characteristics_global']
    if characteristics_global is not None:
        df = df_original.copy()
        # Get the values of characteristics-global from the first line
        characteristics_dict['characteristics_global'] = df[characteristics_global].head(1)

    # Characteristics
    characteristics = task_feature_def_dict[task_type_id]['characteristics']
    characteristics_expected_line_count = task_feature_def_dict[task_type_id]['characteristics_expected_line_count']
    if characteristics is not None:
        df = df_original.copy()
        if df.shape[0] != characteristics_expected_line_count:
            #raise Exception(f'The number of lines in the file {file_name}:{df.shape[0]} is not the same as the number of expected lines: {characteristics_expected_line_count}!')
            print(f'The number of lines in the file {subject_characteristics_file_path}:{df.shape[0]} is not the same as the number of expected lines: {characteristics_expected_line_count}!')

        # Compute the mean of the characteristics
        df = df[characteristics]
        if summarize_characteristics_by_mean:
            df = pd.DataFrame(df[characteristics].mean(), columns=['mean']).transpose()

        # Get the values of characteristics
        characteristics_dict['characteristics'] = df
        
    # AOIs characteristics
    aoi_names = task_feature_def_dict[task_type_id]['aoi_names']
    aoi_characteristics = task_feature_def_dict[task_type_id]['aoi_characteristics']
    if (aoi_names is not None) and (aoi_characteristics is not None):
        df = df_original.copy()
        # Retain lines containing only the AOIs of the task
        df = df[df[aoi_id_col_name].isin(aoi_names)]
        # Check if the number of AOIs in the file is the same as the number of AOIs in the task
        if (df.shape[0] != len(aoi_names)):
            raise Exception(f'The number of AOIs in the file {subject_characteristics_file_path} is not the same as the number of AOIs in the task {task_type_id}.')
        # Sort the lines by the AOI name
        df = df.sort_values(by=aoi_id_col_name)
        df = df.reset_index(drop=True)
        # Get the values of AOI characteristics
        characteristics_dict['AOIs_characteristics'] = df[aoi_characteristics]

    # Determine whether there are missing values in the data
    for df in [characteristics_dict['characteristics_global'], characteristics_dict['characteristics'], characteristics_dict['AOIs_characteristics']]:
        if df is not None:
            missing_values = df.isnull().values.any()
            if missing_values:
                print(f'  - {subject_characteristics_file_path}, missing values: {missing_values}')
    
    return subject_id, characteristics_dict

def create_subject_characteristics_profile(subject_id, characteristics_dict):
    subject_dfs = []

    # Create a new data frame with the subject id
    subject_dfs.append(pd.DataFrame({'subject_id': [subject_id]}))

    for df in [characteristics_dict['characteristics_global'], characteristics_dict['characteristics'], characteristics_dict['AOIs_characteristics']]:
        if df is not None:   
            df_flattened = df.unstack().to_frame().sort_index(level=1).T
            subject_dfs.append(df_flattened)
    
    return pd.concat(subject_dfs, axis=1)

class FixationImageTransform(transforms.Compose):
    def __init__(self):
        fixation_image_transforms = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        super().__init__(fixation_image_transforms)

def generate_fixation_image(df_fixations, degrees_visual_angle_pixels, fixation_duration_color_norm, x_min, x_max, y_min, y_max):

    # Plot the fixations
    default_dpi = 100.0
    fig, ax = plt.subplots(dpi=default_dpi, figsize=((x_max - x_min + 1) / default_dpi, (y_max - y_min + 1) / default_dpi))
    for x, y, w, h, d in df_fixations.values:
        ellipse = Ellipse(xy=(x, y), width=w*degrees_visual_angle_pixels, height=h*degrees_visual_angle_pixels)
        ellipse.set_clip_box(ax.bbox)
        ellipse.set_alpha(0.5)
        ellipse.set_facecolor(plt.colormaps['hot'](fixation_duration_color_norm(d)))
        ax.add_patch(ellipse)

    plt.box(False)
    plt.axis('equal')
    plt.axis('tight')
    plt.tight_layout()
    plt.margins(0, 0)
    ax.set_ylim(y_max, y_min) # Set the y-axis in the opposite direction
    ax.set_xlim(x_min, x_max)
    plt.axis("off")
    #ax.set_facecolor("yellow")

    #plt.plot()
    #plt.show()
    return fig

def save_fixation_image(fig, out_data_fixation_images_dir, task_id, subject_id, trial_id):
    fig_dir = os.path.join(out_data_fixation_images_dir, task_id)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.savefig(os.path.join(fig_dir, subject_id + '_' + str(trial_id) + '.png'), bbox_inches='tight', facecolor='white', transparent=False, pad_inches=0)
    plt.close(fig)

def generate_subject_fixation_images(generate_fixation_image_for_each_trialid, fixation_image_characteristics_names, fill_null_values_by_zero, subject_id_col_name, degrees_visual_angle_pixels, fixation_duration_color_norm, x_min, x_max, y_min, y_max, subject_fixations_file_path):

    figs_dict = {}
    df_fixations_all = []

    df_original = pd.read_csv(subject_fixations_file_path, skiprows=0, sep=',')

    # Fill the missing (nan) values with 0
    if fill_null_values_by_zero:
        df_original = df_original.fillna(0.0)

    # Get the subject id from the first line
    subject_id = str(df_original[subject_id_col_name].iloc[0])

    # For each distint value in 'trialid' column, create a separate fixation image
    if generate_fixation_image_for_each_trialid:
        for trial_id in df_original['trialid'].unique():
            df_fixations = df_original[df_original['trialid'] == trial_id][fixation_image_characteristics_names]
            df_fixations_all.append(df_fixations)
            figs_dict[trial_id] = generate_fixation_image(df_fixations, degrees_visual_angle_pixels, fixation_duration_color_norm, x_min, x_max, y_min, y_max)
    else:
        trial_id = -1
        df_fixations = df_original[fixation_image_characteristics_names]
        df_fixations_all.append(df_fixations)
        figs_dict[trial_id] = generate_fixation_image(df_fixations, degrees_visual_angle_pixels, fixation_duration_color_norm, x_min, x_max, y_min, y_max)
    
    return subject_id, figs_dict, df_fixations_all
