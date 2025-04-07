import copy
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torchvision.transforms as transforms

def build_global_statistics_data_loader(data, labels, train_ids, test_ids, batch_size=128):
    scaler = StandardScaler()

    train_dataset = TensorDataset(torch.from_numpy(scaler.fit_transform(data.loc[train_ids].values)), torch.from_numpy(labels[train_ids]))
    test_dataset = TensorDataset(torch.from_numpy(scaler.transform(data.loc[test_ids].values)), torch.from_numpy(labels[test_ids]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, scaler


def build_fixation_visualisation_data_loader(image_paths, image_class_ids, train_ids, test_ids, batch_size=128, dataset=None):
    if dataset is None:
        return None, None
    
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_ids), batch_size=batch_size)

    return train_loader, test_loader, None

def create_image_folder_dataset(image_paths, image_class_ids):
    return ImageFolderDataset(
        image_paths,
        image_class_ids,
        transform=transforms.Compose(
           [
               transforms.Resize((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
           ]
        )
    )

class ImageFolderDataset(Dataset):

    def __init__(self, image_paths, image_class_ids, transform=None):
        self.image_paths = image_paths
        self.image_class_ids = image_class_ids
        self.transform = transform
    
    def __getitem__(self, index):
        try:
            img = Image.open(self.image_paths[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        except Exception as ex:
            img = None
            print(f'Error while processing file: {self.image_paths[index]} {ex}', file=sys.stderr)
        
        return img, self.image_class_ids[index]
    
    def __len__(self):
        return len(self.image_paths)

def save_sklearn_classifier(exp_id, out_models_dir, classifier):
    model_file = open(os.path.join(out_models_dir, f'{exp_id}.scikit-learn.pkl'), 'wb')
    pickle.dump(classifier, model_file)
    model_file.close()

def load_sklearn_classifier(models_dir, model_file_name):
    return pickle.load(open(os.path.join(models_dir, model_file_name), 'rb'))

def get_sklearn_model_file_name(exp_id):
    return f'{exp_id}.scikit-learn.pkl'

def get_cross_validate_pytorch_model_path(out_models_dir, exp_id, epoch_n, lr):
    return os.path.join(out_models_dir, f'{exp_id}-e{epoch_n}_lr{lr}.pt')

def cross_validate_sklearn(
    exp_id,
    out_models_dir,
    out_results_dir,
    build_classifier,
    data: pd.DataFrame,
    labels: pd.DataFrame,
    df_subject_ids: pd.DataFrame,
    df_class_labels: pd.DataFrame,
    cross_validation,
    scoring = balanced_accuracy_score,
    use_scaler=True
):
    torch.manual_seed(42)
    
    cv_scores = []
    total_preds_df = pd.DataFrame()
    for train_ids, test_ids in cross_validation.split(labels, labels):
        train_data = data.iloc[train_ids].values
        test_data = data.iloc[test_ids].values

        # Scale the data
        if use_scaler:
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)

        fold_classifier = build_classifier()
        fold_classifier.fit(train_data, labels[train_ids])

        # Save the model
        save_sklearn_classifier(exp_id, out_models_dir, fold_classifier)
        # Save the scaler
        if use_scaler:
            scaler_file = open(os.path.join(out_models_dir, f'{exp_id}.scaler.pkl'), 'wb')
            pickle.dump(scaler, scaler_file)
            scaler_file.close()

        # Classification
        total_probs = torch.from_numpy(fold_classifier.predict_proba(test_data))
        total_probs, total_preds = torch.max(total_probs, 1)
        total_targets = labels[test_ids]
        cv_acc = scoring(np.asarray(total_targets), np.asarray(total_preds))
        cv_scores.append(cv_acc)

        # Create a dataframe with the subject ids and the predicted class
        preds_df = pd.DataFrame(data={'subject_id': test_ids, 'subject_class_id': total_targets, 'pred_class_id': total_preds, 'pred_prob': total_probs})
        
        # Replace the relative subject_id with the original subject_id
        preds_df['subject_id'] = df_subject_ids.iloc[preds_df['subject_id']].values

        df_class_labels_target = df_class_labels.copy()
        df_class_labels_target.columns = ['subject_class_id', 'subject_class_label']
        preds_df = preds_df.merge(df_class_labels_target, on='subject_class_id', how='left')

        df_class_labels_pred = df_class_labels.copy()
        df_class_labels_pred.columns = ['pred_class_id', 'pred_class_label']
        preds_df = preds_df.merge(df_class_labels_pred, on='pred_class_id', how='left')

        # Change order of columns
        preds_df = preds_df[['subject_id', 'subject_class_id', 'subject_class_label', 'pred_class_id', 'pred_class_label', 'pred_prob']]
        
        # Append the predictions to the total predictions
        total_preds_df = pd.concat([total_preds_df, preds_df])

    total_preds_df.to_csv(os.path.join(out_results_dir, f'{exp_id}.csv'), index=False)
    return np.array(cv_scores)

def cross_validate_pytorch(
    exp_id,
    out_models_dir,
    out_results_dir,
    build_classifier,
    epoch_n,
    lr: float,
    data, #: pd.DataFrame,
    labels, #: pd.DataFrame,
    df_subject_ids: pd.DataFrame,
    df_class_labels: pd.DataFrame,
    data_loader_gen,
    cross_validation,
    train_classifier = True,
    sigmoid = False,
    scoring = balanced_accuracy_score,
    **data_loader_params,
):
    torch.manual_seed(42)

    log_file = open(os.path.join(out_models_dir, f'{exp_id}-e{epoch_n}_lr{lr}.log'), 'a')

    # GPU support
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.manual_seed(42)

    total_preds_df = pd.DataFrame()
    cv_scores = []
    cv_score_best = -1.0
    best_model_scaler = None
    best_model_state_dict = None
    best_model_desc = ''
    cv_split_id = 0
    for train_ids, test_ids in cross_validation.split(labels, labels):
        cv_split_id += 1
        log_file.write(f'CV_split {cv_split_id} (df_subject_ids.row_count={df_subject_ids.shape[0]}):\n')
        log_file.write(f'* train_pos_ids:{train_ids}|train_subject_ids:{df_subject_ids.iloc[train_ids].values}\n')
        log_file.write(f'* test_pos_ids:{test_ids}|test_subject_ids:{df_subject_ids.iloc[test_ids].values}\n')

        fold_score_best_epoch = -1.0
        fold_best_model_state_dict = None
        fold_best_model_desc = ''
        train_loader, test_loader, scaler = data_loader_gen(data, labels, train_ids, test_ids, batch_size=128, **data_loader_params)
        
        classifier = build_classifier()
        classifier = classifier.to(device)

        if train_classifier:
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
            loss_function = nn.BCEWithLogitsLoss() if sigmoid else nn.CrossEntropyLoss()

            for epoch_no in range(epoch_n):

                # Training
                classifier.train()
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = classifier(inputs)
                    loss = loss_function(outputs, targets.unsqueeze(1) if sigmoid else targets)

                    loss.backward()
                    optimizer.step()

                # Validation
                classifier.eval()
                with torch.no_grad():
                    epoch_val_probs = []
                    epoch_val_preds = []
                    epoch_val_targets = []
                    for inputs, targets in test_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        outputs = classifier(inputs).cpu()
                        outputs = outputs.softmax(dim=1)
                        probs, preds = torch.max(outputs, 1)
                        epoch_val_probs.append(probs)
                        epoch_val_preds.append(preds)
                        epoch_val_targets.append(targets.cpu())
                    
                    epoch_val_probs = np.concatenate(epoch_val_probs)
                    epoch_val_preds = np.concatenate(epoch_val_preds)
                    epoch_val_targets = np.concatenate(epoch_val_targets)
                    epoch_val_acc = scoring(np.asarray(epoch_val_targets), np.asarray(epoch_val_preds))

                log_file.write(f'epoch {epoch_no}: {epoch_val_acc:.4f}\n')
                
                # Best model for the fold
                if epoch_val_acc > fold_score_best_epoch:
                    fold_score_best_epoch = epoch_val_acc
                    fold_best_model_state_dict = copy.deepcopy(classifier.state_dict())
                    fold_best_model_desc = f'CV_split={cv_split_id},epoch={epoch_no},acc={epoch_val_acc:.4f}'

                # Best model globally over all cross-validation folds
                if epoch_val_acc > cv_score_best:
                    cv_score_best = epoch_val_acc
                    best_model_scaler = scaler
                    best_model_state_dict = copy.deepcopy(classifier.state_dict())
                    best_model_desc = f'CV_split={cv_split_id},epoch={epoch_no},acc={epoch_val_acc:.4f}'

        # Re-Validation on the best cross-validation model
        fold_best_model = build_classifier()
        fold_best_model.load_state_dict(fold_best_model_state_dict)
        fold_best_model = fold_best_model.to(device)
        total_probs = []
        total_preds = []
        total_targets = []

        fold_best_model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = fold_best_model(inputs).cpu()
                outputs = outputs.softmax(dim=1)
                probs, preds = torch.max(outputs, 1)
                total_probs.append(probs)
                total_preds.append(preds)
                total_targets.append(targets.cpu())
            
        total_probs = np.concatenate(total_probs)
        total_preds = np.concatenate(total_preds)
        total_targets = np.concatenate(total_targets)

        cv_acc = scoring(np.asarray(total_targets), np.asarray(total_preds))
        cv_scores.append(cv_acc)

        log_file.write(f'Fold best model ({fold_best_model_desc}): {cv_acc:.4f}\n')
        
        # Create a dataframe with the subject ids and the predicted class
        preds_df = pd.DataFrame(data={'subject_id': test_ids, 'subject_class_id': total_targets, 'pred_class_id': total_preds, 'pred_prob': total_probs})
        
        # Replace the relative subject_id with the original subject_id
        preds_df['subject_id'] = df_subject_ids.iloc[preds_df['subject_id']].values

        df_class_labels_target = df_class_labels.copy()
        df_class_labels_target.columns = ['subject_class_id', 'subject_class_label']
        preds_df = preds_df.merge(df_class_labels_target, on='subject_class_id', how='left')

        df_class_labels_pred = df_class_labels.copy()
        df_class_labels_pred.columns = ['pred_class_id', 'pred_class_label']
        preds_df = preds_df.merge(df_class_labels_pred, on='pred_class_id', how='left')

        # Change order of columns
        preds_df = preds_df[['subject_id', 'subject_class_id', 'subject_class_label', 'pred_class_id', 'pred_class_label', 'pred_prob']]
        
        # Append the predictions to the total predictions
        total_preds_df = pd.concat([total_preds_df, preds_df])

    # Save the best model (globally best over all the cross-validation folds)
    torch.save(best_model_state_dict, get_cross_validate_pytorch_model_path(out_models_dir, exp_id, epoch_n, lr))
    # Save the scaler of the best model
    scaler_file_path = os.path.join(out_models_dir, f'{exp_id}-e{epoch_n}_lr{lr}.scaler.pkl')
    if best_model_scaler is None:
        if os.path.exists(scaler_file_path):
            os.remove(scaler_file_path)
    else:
        scaler_file = open(scaler_file_path, 'wb')
        pickle.dump(best_model_scaler, scaler_file)
        scaler_file.close()

    log_file.write(f'Best model ({best_model_desc}): {cv_score_best:.4f}\n')
    log_file.close()

    total_preds_df.to_csv(os.path.join(out_results_dir, f'{exp_id}-e{epoch_n}_lr{lr}.csv'), index=False)
    return np.array(cv_scores), cv_score_best
