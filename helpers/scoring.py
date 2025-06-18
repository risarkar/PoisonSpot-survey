
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from captum.attr import IntegratedGradients
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as prf
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.dnn import DNN

__all__ = [
    "train_prov_data_custom",
    "score_poisoned_samples",
    "get_diff",
]

def train_prov_data_custom(
    X_sus,
    X_clean,
    clean_igs_inds,
    sus_igs_inds,
    random_poison_idx,
    random_clean_sus_idx,
    n_groups,
    seed,
    device,
    model_name='RandomForest',
    verbose=True,
    training_mode=True,
    max_iters=1,
    confidence_threshold=0.7
):
    """
    Train a provenance-based classifier on combined clean and suspect data groups.

    Args:
        X_sus (np.ndarray or torch.Tensor): Features of suspect (possibly poisoned) samples.
        X_clean (np.ndarray or torch.Tensor): Features of known clean samples.
        clean_igs_inds (list[int]): Indices of clean samples grouped by IG scores.
        sus_igs_inds (list[int]): Indices of suspect samples grouped by IG scores.
        random_poison_idx (list[int]): Randomly selected indices of poisoned samples.
        random_clean_sus_idx (list[int]): Randomly selected indices combining clean and suspect samples.
        n_groups (int): Number of groups for provenance segmentation.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device for any torch-based computation.
        model_name (str, optional): Classifier name, e.g., 'RandomForest' or 'GradientBoosting'. Defaults to 'RandomForest'.
        verbose (bool, optional): Flag to print training progress. Defaults to True.
        training_mode (bool, optional): If True, retrain the model; else only score. Defaults to True.
        max_iters (int, optional): Number of training iterations. Defaults to 1.
        confidence_threshold (float, optional): Minimum confidence for positive predictions. Defaults to 0.7.

    Returns:
        model: Trained classifier instance.
        dict: Performance metrics (e.g., accuracy, precision, recall) for each group.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    y_sus = np.ones(len(X_sus))
    y_clean = np.zeros(len(X_clean))

    X = np.concatenate([X_clean, X_sus])
    del X_sus, X_clean
    y = np.concatenate([y_clean, y_sus])
    assert not np.isinf(X).any()
    del y_clean, y_sus

    def split_images_into_groups(image_indices, n_splits, seed=seed):
        """
        Randomly splits image indices into groups.

        :param image_indices: Array of image indices to split.
        :param n_splits: Number of groups to split into.
        :param seed: Seed for the random number generator.
        :return: Array of image index groups.
        """
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(image_indices)
        return np.array_split(shuffled_indices, n_splits)
    
    
    

    unique_sus_images = np.unique(sus_igs_inds)
    unique_clean_images = np.unique(clean_igs_inds)

    sus_image_groups = split_images_into_groups(unique_sus_images, n_groups)
    clean_image_groups = split_images_into_groups(unique_clean_images, n_groups)

    predictions, true_labels, predictions_proba = [], [], []
    group_feature_importances, index_tracker = [], []

    predictions_with_indices = {}

    concated_igs = np.concatenate([clean_igs_inds, sus_igs_inds])
    del clean_igs_inds, sus_igs_inds

    # Define models
    models = {
        'prf': prf(n_estimators=100, bootstrap=True, n_jobs=-1),
        'RandomForest': RandomForestClassifier(random_state=seed, n_jobs=-1, n_estimators=100, class_weight='balanced'), 
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed),
        'LinearSVM': SVC(kernel='linear', probability=True, random_state=seed),
        'KernelSVM': SVC(kernel='rbf', probability=True, random_state=seed)
    }
    
    
    param_grid = {
        'n_estimators': [100,  300, 500],
        'max_depth': [None, 10, 5],  
    }

    # Iterate through each group to perform cross-validation
    for i in range(n_groups):
        test_sus_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in sus_image_groups[i]])
        train_sus_indices = np.concatenate([np.where(concated_igs  == img_idx)[0] for j, group in enumerate(sus_image_groups) if j != i for img_idx in group])
        
        test_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for img_idx in clean_image_groups[i]])
        train_clean_indices = np.concatenate([np.where(concated_igs == img_idx)[0] for j, group in enumerate(clean_image_groups) if j != i for img_idx in group]) 
        
        # Create training and testing sets
        train_indices = np.concatenate([train_clean_indices, train_sus_indices])
        test_indices = np.concatenate([test_clean_indices, test_sus_indices])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train and evaluate the selected model
        X_labeled = np.empty((0, X_train.shape[1]))  
        y_labeled = np.empty((0,))
        

        pos_inds_train = [i for i, ind in enumerate(concated_igs[train_indices[y_train==1]]) if ind in random_poison_idx]
        pos_inds = np.array([ind for ind in concated_igs[train_indices[y_train==1]]])

        max_iters = 1
        # confidence_threshold = 0.5
        m = 3

        if model_name in models:
            if model_name == "RandomForest":  
                clf = models[model_name]
                iteration = 0
                X_sus_temp = X_train[y_train == 1].copy()
                X_clean_temp = X_train[y_train == 0].copy()

                
                # if i == 0:
                #     grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grid, 
                #                     scoring='f1_weighted', cv=3)
                #     grid_search.fit(X_train, y_train)
                #     clf = grid_search.best_estimator_
                #     print(grid_search.best_params_)
                # else:
                best_params = {'max_depth':10, 'n_estimators': 200}
                clf = RandomForestClassifier(**best_params, n_jobs=-1, random_state=seed)
                # clf = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1, random_state=seed)
                clf.fit(X_train, y_train) 
                
                while iteration < max_iters:
                    iteration += 1
                    
                    if iteration > 1:
                        confidence_threshold = 0.7

                    y_proba = clf.predict_proba(X_sus_temp)[:, 1]
                    
                    # high_conf_indices = np.where(y_proba > confidence_threshold)[0]
                    high_conf_indices = pos_inds_train

                    true_pos = set(pos_inds_train) & set(high_conf_indices)

                    if verbose:
                        print(f"Iteration {iteration}: {len(high_conf_indices)} high-conf , true pos {len(true_pos)}, total pos {len(pos_inds_train)} total sus {len(X_sus_temp)}")

                    if len(high_conf_indices) == 0:
                        break

                    X_labeled = np.concatenate([X_labeled, X_sus_temp[high_conf_indices]])                                                                                                      
                    y_labeled = np.concatenate([y_labeled, np.ones(len(high_conf_indices))])

                    X_train_temp = np.concatenate([X_labeled, X_clean_temp[:len(X_labeled)*m]])
                    y_train_temp = np.concatenate([y_labeled, (len(X_train_temp) - len(X_labeled)) * [0]])                 

                    X_sus_temp = np.delete(X_sus_temp, high_conf_indices, axis=0)  
                    if len(X_sus_temp) == 0:
                        break
                    remaining_indices = set(range(len(X_sus_temp))) - set(high_conf_indices)
                    pos_inds_train = [i for i, ind in enumerate(remaining_indices) if pos_inds[ind] in random_poison_idx]
                    pos_inds  = np.delete(pos_inds, high_conf_indices , axis=0)
                    clf.fit(X_train_temp, y_train_temp)     

 
            else:
                if i == 0:
                    grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grid, 
                                    scoring='f1_weighted', cv=2)
                    grid_search.fit(X_train, y_train)
                    clf = grid_search.best_estimator_
                    print(grid_search.best_params_)
                else:
                    clf = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1, random_state=seed)
                    clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            if model_name in ['RandomForest', 'LogisticRegression', 'LinearSVM', 'prf']:
                feature_importances = clf.coef_[0] if model_name in ['LogisticRegression', 'LinearSVM'] else clf.feature_importances_
                group_feature_importances.append(feature_importances)
            else:
                # Use permutation importance for kernel SVM
                perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=seed)
                group_feature_importances.append(perm_importance.importances_mean)
        elif model_name == 'MLP':
            # Train MLP
            input_shape = X_train.shape[1]
            model = DNN(input_shape, 2)
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)    

            X_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            X_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

            X_train = DataLoader(X_train, batch_size=5000, shuffle=True)
            X_test = DataLoader(X_test, batch_size=5000, shuffle=False)

            model.train(mode = training_mode)
            for epoch in range(10):
                for inputs, labels in X_train:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if np.isnan(inputs.cpu()).any():
                        inputs = np.nan_to_num(inputs.cpu(), nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
                        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            model.eval()
            y_pred, y_pred_proba = [], []
            with torch.no_grad():
                for inputs, labels in X_test:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if np.isnan(inputs.cpu()).any():
                        inputs = np.nan_to_num(inputs.cpu(), nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
                        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    y_pred.extend(preds.cpu().numpy())
                    y_pred_proba.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            
            # Calculate feature importance using integrated gradients
            ig = IntegratedGradients(model)
            input_tensor = torch.tensor(X[train_indices][:128], dtype=torch.float32, requires_grad=True).to(device)
            attr, delta = ig.attribute(input_tensor, target=1, return_convergence_delta=True)
            feature_importances = attr.mean(dim=0).detach().cpu().numpy()
            group_feature_importances.append(feature_importances)
        else:
            raise ValueError(f"Model {model_name} is not supported")

        predictions.extend(y_pred)
        true_labels.extend(y_test)
        predictions_proba.extend(y_pred_proba)
        index_tracker.extend(test_indices)

        TPR = recall_score(y_test, y_pred)
        ACC = accuracy_score(y_test, y_pred)
        pos_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_poison_idx]
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        pos_recall = recall_score(y_test[pos_inds], y_pred[pos_inds])
        
        if verbose:
            sus_clean_inds = [i for i, ind in enumerate(concated_igs[test_indices]) if ind in random_clean_sus_idx]
            if len(sus_clean_inds) > 0:
                sus_clean_acc = recall_score(y_test[sus_clean_inds], y_pred[sus_clean_inds])
                print(f"Model: {model_name} - Group {i+1} Test Acc: {ACC:.4f} TPR: {TPR:.4f} Poison TPR: {pos_recall:.4f} Suspected Clean TPR: {sus_clean_acc:.4f}")
            else:
                print(f"Model: {model_name} - Group {i+1} Test TPR: {TPR:.4f} Acc: {ACC:.4f}")
        
        del X_train, X_test, y_train, y_test
        for idx, pred in zip(test_indices, y_pred_proba):
            if concated_igs[idx] in predictions_with_indices:
                predictions_with_indices[concated_igs[idx]].append(pred)
            else: 
                predictions_with_indices[concated_igs[idx]] = [pred]
    
    final_acc = accuracy_score(true_labels, predictions)
    final_tpr = recall_score(true_labels, predictions)
    if verbose:
        print(f"Final TPR: {final_tpr} Final Acc: {final_acc}")

    average_feature_importances = np.mean(group_feature_importances, axis=0)
    
    true_labels = np.array(true_labels)
    predictions_proba = np.array(predictions_proba)
    index_tracker = np.array(index_tracker)
    
    return predictions_with_indices, average_feature_importances, true_labels, predictions_proba, final_acc, index_tracker
 
 
 

def score_poisoned_samples(
    sus_diff,
    clean_diff,
    clean_inds,
    sus_inds,
    poison_indices,
    random_clean_sus_idx,
    n_groups,
    dataset,
    cv_model,
    epochs,
    seed,
    device,
    poison_ratio,
    percentage,
    attack,
    figure_path,
    threshold=0.6,
    threshold_type = "Kmeans",
    k_2 = 0.0001  
):
    """
    Score and visualize the likelihood of samples being poisoned based on weight update differences.

    Args:
        sus_diff (np.ndarray): Weight update differences for suspect samples.
        clean_diff (np.ndarray): Weight update differences for clean samples.
        clean_inds (list[int]): Indices of clean samples used for scoring.
        sus_inds (list[int]): Indices of suspect samples used for scoring.
        poison_indices (list[int]): Known indices of poisoned samples for benchmarking.
        random_clean_sus_idx (list[int]): Random index selection combining clean and suspect.
        n_groups (int): Number of provenance groups considered.
        dataset (str): Name of the dataset (e.g., 'CIFAR10').
        cv_model: Pretrained or cross-validated model for scoring.
        epochs (int): Number of epochs used during weight update capture.
        seed (int): Random seed for reproducibility.
        device (torch.device): Device for computation.
        poison_ratio (float): Ratio of poisoned samples in training.
        percentage (float): Percentage threshold for group selection.
        attack (str): Backdoor attack identifier.
        figure_path (str): Path to save result figures.
        threshold (float, optional): Decision threshold for labeling a sample as poisoned. Defaults to 0.6.

    Returns:
        pd.DataFrame: Table of sample scores and predicted labels.
        dict: Summary statistics (true positives, false positives, etc.).
    """
    _, average_feature_importances, _, _, _, _ = train_prov_data_custom(
        sus_diff, clean_diff, clean_inds, sus_inds, poison_indices, random_clean_sus_idx, n_groups, seed, device, model_name=cv_model)
    real_clean_indices = np.unique(clean_inds)
    average_original_feature_importances = average_feature_importances  
    
    plt.figure()
    plt.plot(range(len(average_original_feature_importances)), average_original_feature_importances, label='Feature Importances', alpha=0.5, color='blue')
    plt.savefig(figure_path + f"/Feature_importances.png")
    outliers = np.where(average_original_feature_importances > k_2)[0]
    print("len outliers: ", len(outliers))  
    best_rel_feature = len(outliers)
    
    relevant_features = np.argsort(average_original_feature_importances)[::-1][:best_rel_feature]
    
    if len(relevant_features) == 0:
        return [], 0, 0, 0, 0
    predictions_with_indices_2, _, _, _, _, _ = train_prov_data_custom(
            sus_diff[:,relevant_features], clean_diff[:,relevant_features], clean_inds, sus_inds, poison_indices,random_clean_sus_idx, n_groups, seed, device, model_name=cv_model) 
    
        
    pos_predictions_real = []
    clean_predictions_real = []
    sus_clean_predictions = []

    pos_prediction_indices = []
    clean_predicion_indices = []
    sus_clean_prediction_indices = []
    
    if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
        poison_indices_all = poison_indices
        poison_indices = np.concatenate(list(poison_indices_all.values()))

    epoch_num = 1000  
    for k, v in predictions_with_indices_2.items():
        if k in poison_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            pos_predictions_real.append(v)
            pos_prediction_indices.append(k)
        elif k in real_clean_indices:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            clean_predictions_real.append(v)
            clean_predicion_indices.append(k)
        else:
            if len(v) < epoch_num:
                v = np.pad(v, (0, epoch_num - len(v)), mode='constant', constant_values=np.nan)
            sus_clean_predictions.append(v)
            sus_clean_prediction_indices.append(k)
            
    
    pos_predictions_real = np.array(pos_predictions_real)
    clean_predictions_real = np.array(clean_predictions_real)
    sus_clean_predictions = np.array(sus_clean_predictions)

    pos_prediction_indices = np.array(pos_prediction_indices)
    clean_prediction_indices = np.array(clean_predicion_indices)
    sus_clean_prediction_indices = np.array(sus_clean_prediction_indices)

    

    def compute_thresholds(pos_scores, clean_scores, sus_scores):
        if sus_scores is not None:
            combined_data = np.concatenate([pos_scores, sus_scores]).reshape(-1, 1)
        else:
            combined_data = np.concatenate([pos_scores, clean_scores]).reshape(-1, 1)

        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(combined_data)
        
        

        # Compute the GMM Threshold
        x = np.linspace(combined_data.min(), combined_data.max(), num=1000).reshape(-1, 1)
        pdf_individual = gmm.predict_proba(x) * np.exp(gmm.score_samples(x).reshape(-1, 1))
        diff_sign = np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))

        gaussian_threshold = next(x[i, 0] for i in range(1, len(x)) if np.diff(np.sign(pdf_individual[:, 0] - pdf_individual[:, 1]))[i] != 0)
        # Compute the KMeans Threshold
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(combined_data)
        labels = kmeans.labels_
        cluster_0_points = combined_data[labels == 0]
        cluster_1_points = combined_data[labels == 1]
        boundary_points = [min(cluster_0_points), max(cluster_1_points)]
        kmeans_threshold = np.mean(boundary_points)

        # Compute the Outlier Threshold
        mean_score = np.mean(combined_data)
        std_score = np.std(combined_data)
        outlier_threshold = mean_score + 2 * std_score 
        
        print("TPR gaussian threshold: ", np.mean(pos_scores > gaussian_threshold))
        print("TPR kmeans: ", np.mean(pos_scores > kmeans_threshold))

        tpr_kmeans = np.mean(pos_scores > kmeans_threshold)
        tpr_gaussian = np.mean(pos_scores > gaussian_threshold)
        
        if sus_scores is not None:
            print("FPR gaussian: ", np.mean(sus_scores > gaussian_threshold))
            print("FPR kmeans: ", np.mean(sus_scores > kmeans_threshold))
            fpr_kmeans = np.mean(sus_scores > kmeans_threshold)
            fpr_gaussian = np.mean(sus_scores > gaussian_threshold)
        else:
            print("FPR gaussian: ", np.mean(clean_scores > gaussian_threshold))
            print("FPR kmeans: ", np.mean(clean_scores > kmeans_threshold))

            fpr_kmeans = np.mean(clean_scores > kmeans_threshold)
            fpr_gaussian = np.mean(clean_scores > gaussian_threshold)
            
            

        return threshold, kmeans_threshold, gaussian_threshold, tpr_kmeans, fpr_kmeans, tpr_gaussian, fpr_gaussian 

    def plot_scores(pos_scores, clean_scores, sus_scores, title):
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'orange', 'red']
        plt.rcParams.update({'font.size': 16})
        if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
            start_index = 0
            for key in poison_indices_all:
                attack_indices = [i for i, ind in enumerate(pos_prediction_indices) if ind in poison_indices_all[key]]
                if key == "LabelConsistent":
                    key = "Label-Consistent"
                plt.scatter(np.arange(len(pos_scores[attack_indices])), pos_scores[attack_indices], label=f'Poison {key}', alpha=0.5, color=colors[start_index])
                start_index += 1
        else:
            plt.scatter(np.arange(len(pos_scores)), pos_scores, label='Poison', alpha=0.5, color='red')
        # plt.scatter(np.arange(len(clean_scores)), clean_scores, label='Clean', alpha=0.5, color='blue')
        if sus_scores is not None:
            plt.scatter(np.arange(len(sus_scores)), sus_scores, label='Clean Suspected', alpha=0.5, color='green')

        threshold, kmeans_threshold, gaussian_threshold, tpr_kmeans, fpr_kmeans, tpr_gaussian, fpr_gaussian  = compute_thresholds(pos_scores, clean_scores, sus_scores)
        
        thresholds = [threshold, kmeans_threshold, gaussian_threshold]
        print("thresholds: ", thresholds)
        labels = ['Threshold', 'Threshold 1 (Kmeans)', 'Threshold 2 (Gaussian)']
        colors = ['black', 'green', 'orange']

            
        for thr, color, label in zip(thresholds[1:], colors[1:], labels[1:]):
            plt.axhline(y=thr, color=color, linestyle='--', label=label)

        plt.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, 1.02),  
            ncol=3, 
            borderaxespad=0
        )
        # plt.axhline(y=threshold, color='black', linestyle='--')
        plt.xlabel("Number of samples")
        plt.ylabel("Poisoning Score")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  
        plt.savefig(figure_path + f"{title}.png")


        
        return threshold, kmeans_threshold, gaussian_threshold, tpr_kmeans, fpr_kmeans, tpr_gaussian, fpr_gaussian

    def average_k_minimum_values(arr, k):
        nan_mask = np.isnan(arr)
        large_number = np.nanmax(arr[np.isfinite(arr)]) + 1
        arr_masked = np.where(nan_mask, large_number, arr)
        
        k_min_indices = np.argsort(arr_masked, axis=1)[:, :k]
        
        k_min_values = np.take_along_axis(arr, k_min_indices, axis=1)
        
        k_min_averages = np.mean(k_min_values, axis=1)
        
        return k_min_averages

    j = 50
    pos_mean = np.nanmean(pos_predictions_real, axis=1)
    pos_max = average_k_minimum_values(pos_predictions_real, j)
    pos_mean_max = pos_mean * pos_max

    clean_mean = np.nanmean(clean_predictions_real, axis=1)
    clean_max = average_k_minimum_values(clean_predictions_real, j)
    clean_mean_max = clean_mean * clean_max

    if len(sus_clean_predictions) > 0:
        sus_mean = np.nanmean(sus_clean_predictions, axis=1)
        sus_max = average_k_minimum_values(sus_clean_predictions, j)
        sus_mean_max = sus_mean * sus_max
    else:
        sus_mean = sus_max = sus_mean_max = None

    custom_threshold_mean,kmeans_threshold_mean, gaussian_threshold_mean,  tpr_kmeans, fpr_kmeans, tpr_gaussian, fpr_gaussian = plot_scores(pos_mean, clean_mean, sus_mean, f"/SL Mean {attack} pr {poison_ratio} percentage {percentage} constant threshold")
    if threshold_type == "kmeans":
        best_config = "kmeans"
    elif threshold_type == "gaussian":
        best_config = "gaussian"
    elif threshold_type == "custom":
        best_config = "custom"
    else:
        raise ValueError(f"Threshold type {threshold_type} is not supported")
    
    if attack == "narcissus_lc" or attack == "narcissus_lc_sa":
        for key in poison_indices_all:
            attack_indices = [i for i, ind in enumerate(pos_prediction_indices) if ind in poison_indices_all[key]]
            print(f"TPR {key} kmeans threshold: ", np.mean(pos_mean[attack_indices] > kmeans_threshold_mean))
            print(f"FPR {key} kmeans threshold: ", np.mean(clean_mean[attack_indices] > kmeans_threshold_mean))

                
    if len(sus_clean_prediction_indices) > 0:
        values = {
        "gaussian": np.concatenate([pos_mean, sus_mean]) > gaussian_threshold_mean,
        "kmeans": np.concatenate([pos_mean, sus_mean]) > kmeans_threshold_mean,
        "custom": np.concatenate([pos_mean, sus_mean]) > custom_threshold_mean,
        }
        
        sus_prediction_indices = np.concatenate([pos_prediction_indices, sus_clean_prediction_indices])
        indexes_to_exclude = sus_prediction_indices[values[best_config]]
        return indexes_to_exclude, tpr_kmeans, fpr_kmeans, tpr_gaussian, fpr_gaussian
        
    else:
        values = {
        "gaussian": pos_mean > gaussian_threshold_mean,
        "kmeans": pos_mean > kmeans_threshold_mean,
        "custom": pos_mean > custom_threshold_mean,
        }
        
        print("values: ", len(pos_mean[pos_mean > gaussian_threshold_mean]))
        indexes_to_exclude = pos_prediction_indices[values[best_config]]            
    return indexes_to_exclude, tpr_kmeans, fpr_kmeans, tpr_gaussian, fpr_gaussian


def get_diff(sus_inds, clean_inds, clean_inds_2):
    sus_indices = np.array([
        image_idx
        for epoch in sus_inds
        for image_idx in sus_inds[epoch]
    ])
    sus_array = np.array([
        sus_inds[epoch][image_idx]
        for epoch in sus_inds
        for image_idx in sus_inds[epoch]
    ])

    clean_indices = np.array([
        image_idx[0]
        for epoch in clean_inds
        for image_idx
        in clean_inds[epoch]
    ])
    clean_array= np.array([
        clean_inds[epoch][image_idx]
        for epoch in clean_inds
        for image_idx in clean_inds[epoch]
    ])


    clean_2_array = np.array([
        clean_inds_2[epoch][image_idx]
        for epoch in clean_inds_2
        for image_idx in clean_inds_2[epoch]
    ])


    sus_diff = sus_array - clean_2_array
    clean_diff = clean_array - clean_2_array
    return sus_diff, clean_diff, sus_indices, clean_indices

