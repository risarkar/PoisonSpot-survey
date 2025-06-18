import os
import time
import random
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
from src.utils.util import AverageMeter
from torch.utils.data import DataLoader


__all__ = [
    "capture_first_level_multi_epoch_batch_sample_weight_updates",
    "capture_sample_level_weight_updates_idv",
]


def capture_first_level_multi_epoch_batch_sample_weight_updates(
    random_sus_idx,
    model,
    orig_model,
    optimizer,
    opt,
    scheduler,
    criterion,
    training_epochs,
    lr,
    poisoned_train_loader,
    test_loader,
    poisoned_test_loader,
    target_class,
    sample_from_test,
    attack,
    device,
    seed,
    figure_path,
    training_mode=True,
    k=1
):
    """
    Capture batch-level weight update dynamics over multiple epochs for suspected samples.

    Args:
        random_sus_idx (list[int]): Indices of randomly selected suspect samples.
        model (torch.nn.Module): The current model under training.
        orig_model (torch.nn.Module): The initial, unmodified model state.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        opt (dict): Dictionary of optimizer hyperparameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (callable): Loss function.
        training_epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        poisoned_train_loader (DataLoader): DataLoader for poisoned training data.
        test_loader (DataLoader): DataLoader for clean test data.
        poisoned_test_loader (DataLoader): DataLoader for backdoored test data.
        target_class (int): Target class index for backdoor attack.
        sample_from_test (bool): Whether to sample from test set for evaluation.
        attack (str): Identifier for the attack method.
        device (torch.device): Device on which tensors are allocated.
        seed (int): Random seed for reproducibility.
        figure_path (str): File path to save visualization figures.
        training_mode (bool, optional): If True, run in training mode. Defaults to True.
        k (int, optional): Number of top features or samples to track. Defaults to 1.

    Returns:
        None: Saves weight update visualizations and logs to figure_path.
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []

    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    dataset = poisoned_train_loader.dataset

    
    if not sample_from_test:
        target_images = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] == target_class and dataset[i][2] not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        test_dataset = test_loader.dataset
        target_images = [test_dataset[i][0].clone() for i in range(len(test_dataset)) if test_dataset[i][1] == target_class]
        
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for img in target_images]
        elif attack == 'sa':
            target_images = [transform_sa(img) for img in target_images]
               
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
        print("target images shape:", target_images.shape, "attack: ", attack)
        
    
    
    sur_model = copy.deepcopy(model).to(device)
    if opt == 'sgd':
        sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
    
    len_model_params = len(np.concatenate([model.state_dict()[layer].cpu().numpy().flatten() for layer in model.state_dict()]))
    
    
    
    if not os.path.exists("src/temp_folder"):
        os.makedirs("src/temp_folder")
    
    torch.save(optimizer.state_dict(), f"src/temp_folder/temp_optimizer_state_dict_{random_num}.pth")
    
    sur_optimizer_state = torch.load(f"src/temp_folder/temp_optimizer_state_dict_{random_num}.pth", map_location=device)
    sur_optimizer.load_state_dict(sur_optimizer_state)
    
    os.remove(f"src/temp_folder/temp_optimizer_state_dict_{random_num}.pth")
        
    
    for epoch in tqdm(range(training_epochs)):
        clean_params = np.array([-np.inf] * len_model_params)
        sus_params = np.array([-np.inf] * len_model_params)
        print("Length of clean params: ", len(clean_params))

        # Train
        model.to(device) 
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        
        step1_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
        

        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device).float(), labels.to(device).long(), indices                       
            
            start_time = time.time()
            
            original_weights = copy.deepcopy(model.state_dict())
            
            # model.eval()                     
            model.train(mode = training_mode)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            
            images = images.clone()
            labels = labels.clone()
            indices = indices.clone()
    

            
            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    # print("Length of available indices: ", len(available_indices), "Length of pos indices: ", len(pos_indices), "Length of target indices: ", len(target_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    
                    remaining_indices = list(set(range(len(indices))) - set(pos_indices) - set(clean_indices) - set(clean_indices_2))
                    assert set(indices[remaining_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices]) & set(random_sus_idx) == set() and  set(indices[clean_indices_2]) & set(random_sus_idx) == set()

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    remaining_indices =  list(set(range(len(indices))) - set(pos_indices))
                    assert set(indices[remaining_indices].cpu().numpy()) | set(indices[pos_indices].cpu().numpy()) == set(indices.cpu().numpy()) and set(indices[remaining_indices].cpu().numpy()) & set(indices[pos_indices].cpu().numpy()) == set()
                
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()
                sur_model.train(mode = training_mode)                   

                
                sur_model.load_state_dict(original_weights)
                # sur_optimizer.load_state_dict(sur_optimizer_state)
            
                
                sur_optimizer.zero_grad()
                
                step_4_2_time = time.time() - start_time
                step_4_2_time_avg += step_4_2_time
                start_time = time.time()
                
                
                output = sur_model(images[pos_indices])
                    
                pred_labels = output.argmax(dim=1)
                
                loss = criterion(output, labels[pos_indices])
                loss.backward()
                sur_optimizer.step()
                
                step_4_3_time = time.time() - start_time 
                step_4_3_time_avg += step_4_3_time
                start_time = time.time()
                
                
                
                sur_model_state_dict = sur_model.state_dict()
                temp_sus = torch.cat([
                    (sur_model_state_dict[key] - original_weights[key]).view(-1)
                    for key in sur_model_state_dict
                ]).cpu().numpy()
                    
                
                step4_4_time = time.time() - start_time
                step_4_4_time_avg += step4_4_time
                start_time = time.time()
                
                
                
                sur_model.load_state_dict(original_weights)
                # sur_optimizer.load_state_dict(sur_optimizer_state)
                
                
                sur_optimizer.zero_grad()

                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                    else:
                        clean_batch = images[clean_indices]
                        clean_labels = labels[clean_indices ]
                else:
                    clean_batch = target_images[clean_indices]
                    clean_labels =  torch.tensor([target_class] * len(clean_indices)).to(device)



                sur_model.train(mode = training_mode)  
                output = sur_model(clean_batch)
            
                    
                    
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                temp_clean = torch.cat([
                    (sur_model_state_dict[key] - original_weights[key]).view(-1)
                    for key in sur_model_state_dict
                ]).cpu().numpy()



                step5_time = time.time() - start_time
                start_time = time.time()
                step5_time_avg += step5_time
                
                
                sur_model.load_state_dict(original_weights)
                # sur_optimizer.load_state_dict(sur_optimizer_state)
                
                sur_optimizer.zero_grad()

                start_time_2 = time.time()                
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                        clean_labels = torch.cat([labels[clean_indices_2], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                    else:
                        clean_batch = images[clean_indices_2]
                        clean_labels = labels[clean_indices_2]
                else:
                    clean_batch =  target_images[clean_indices_2]
                    clean_labels = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                

                step5_1_time = time.time() - start_time_2
                step5_1_time_avg += step5_1_time
                
                sur_model.train(mode = training_mode)    
                output = sur_model(clean_batch)
                
                    
                clean_labels = clean_labels.long()
                loss = criterion(output, clean_labels)
                loss.backward()
                sur_optimizer.step()
                
                sur_model_state_dict = sur_model.state_dict()
                temp_clean_2 = torch.cat([
                        (sur_model_state_dict[key] - original_weights[key]).view(-1)
                        for key in sur_model_state_dict.keys()
                ]).cpu().numpy()

            
                
                step6_time = time.time() - start_time   
                step6_time_avg += step6_time 
                start_time = time.time()
                
                sus_params = np.maximum(sus_params, np.abs(temp_sus - temp_clean_2))
                clean_params = np.maximum(clean_params, np.abs(temp_clean - temp_clean_2))
                                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                

        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        
        
        
        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device).float(), labels.to(device).long()
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
        
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:',out_loss)

        differences = np.abs(sus_params - clean_params)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        z_scores = (differences - mean_diff) / std_diff
        cut_num = len(np.where(np.abs(z_scores) > k)[0])
        important_features = np.argsort(np.abs(z_scores))[::-1][:cut_num]
        if epoch == 0:
            important_features_avg = important_features
        else:
            important_features_avg = np.intersect1d(important_features_avg, important_features)
            print("Number of important features after intersection:", len(important_features_avg), "Epoch:", epoch)
        plt.figure()
        plt.scatter(range(sus_params.shape[0]), z_scores, label='Z Scores', alpha=0.5, color='blue')
        plt.savefig(figure_path + f"/Max_diff.png")
        
        plt.figure()
        plt.scatter(range(sus_params.shape[0]), differences, label='Differences', alpha=0.5, color='red')
        plt.savefig(figure_path + f"/differences.png")
    return important_features_avg


def capture_sample_level_weight_updates_idv(
    random_sus_idx,
    model,
    orig_model,
    optimizer,
    opt,
    scheduler,
    criterion,
    training_epochs,
    lr,
    poisoned_train_loader,
    test_loader,
    poisoned_test_loader,
    important_features,
    target_class,
    sample_from_test,
    attack,
    device,
    seed,
    figure_path,
    training_mode=True,
    k=1
):
    """
    Capture individual sample-level weight update trajectories for specified important features.

    Args:
        random_sus_idx (list[int]): Indices of randomly selected suspect samples.
        model (torch.nn.Module): The current model under training.
        orig_model (torch.nn.Module): The initial, unmodified model state.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        opt (dict): Dictionary of optimizer hyperparameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (callable): Loss function.
        training_epochs (int): Number of training epochs.
        lr (float): Initial learning rate.
        poisoned_train_loader (DataLoader): DataLoader for poisoned training data.
        test_loader (DataLoader): DataLoader for clean test data.
        poisoned_test_loader (DataLoader): DataLoader for backdoored test data.
        important_features (list[int]): Indices of features to track individually.
        target_class (int): Target class index for backdoor attack.
        sample_from_test (bool): Whether to sample from test set for evaluation.
        attack (str): Identifier for the attack method.
        device (torch.device): Device on which tensors are allocated.
        seed (int): Random seed for reproducibility.
        figure_path (str): File path to save visualization figures.
        training_mode (bool, optional): If True, run in training mode. Defaults to True.
        k (int, optional): Number of top features or samples to track. Defaults to 1.

    Returns:
        None: Saves weight update visualizations and logs to figure_path.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_ACC = []
    test_ACC = []
    clean_ACC = []

    sus_diff = {}
    clean_diff = {}
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    random_sus_idx = set(random_sus_idx)
    
    dataset = poisoned_train_loader.dataset

    
    
    print("shape of dataset: ", dataset[0][0].shape)
    if not sample_from_test:
        target_images = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == target_class]
        target_images = torch.stack(target_images).to(device)
        target_indices = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] == target_class and dataset[i][2] not in random_sus_idx]
    else:
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ])
        
        transform_sa = Compose([
            RandomHorizontalFlip(),
        ])
        
        test_dataset = test_loader.dataset
        target_images = [test_dataset[i][0].clone() for i in range(len(test_dataset)) if test_dataset[i][1] == target_class]
        if attack == 'narcissus' or attack == 'lc':
            target_images = [transform_train(img) for img in target_images]
        elif attack == 'sa':
            target_images = [transform_sa(img) for img in target_images]
               
            
        print(len(target_images))
        target_images = torch.stack(target_images).to(device)
        print("target images shape:", target_images.shape)
    
    
    separate_rng = np.random.default_rng()
    random_num = separate_rng.integers(1, 10000)
    
    sur_model = copy.deepcopy(model)
    sur_model.to(device)
    if opt == 'sgd':
        sur_optimizer = torch.optim.SGD(params=sur_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        sur_optimizer = torch.optim.Adam(params=sur_model.parameters(), lr=lr)
    
    if not os.path.exists("src/temp_folder"):
        os.makedirs("src/temp_folder")
    torch.save(sur_optimizer.state_dict(), f'src/temp_folder/temp_optimizer_state_dict_{random_num}.pth')
    optimizer_state = torch.load(f'src/temp_folder/temp_optimizer_state_dict_{random_num}.pth')
    sur_optimizer.load_state_dict(optimizer_state)
    
    os.remove(f"src/temp_folder/temp_optimizer_state_dict_{random_num}.pth")
    
        
    model.to(device) 
    for epoch in tqdm(range(training_epochs)):
        
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 

        step1_time_avg = 0
        step2_time_avg = 0
        step3_time_avg = 0
        step_4_1_time_avg = 0
        step_4_2_time_avg = 0
        step_4_3_time_avg = 0
        step_4_4_time_avg = 0
        step_4_5_time_avg = 0
        step5_time_avg = 0
        step5_1_time_avg = 0
        step6_time_avg = 0
        step7_time_avg = 0
    
        
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device).float(), labels.to(device).long(), indices                      
            
            start_time = time.time()
            
            original_weights = copy.deepcopy(model.state_dict())

            step1_time = time.time() - start_time
            start_time = time.time()
            step1_time_avg += step1_time
            

            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg*100, loss_meter.avg))
            
            
          
            
            temp_sus = {}
            temp_clean = {}
            
            
            indices = indices.clone()
            labels  = labels.clone()
            indices = indices.clone()
            

            pos_indices = [i for i, ind in enumerate(indices) if ind.item() in random_sus_idx]
            assert np.all(labels[pos_indices].cpu().numpy() == target_class)
            if len(pos_indices) > 0:
                step2_time = time.time() - start_time
                start_time = time.time()
                step2_time_avg += step2_time
                
                if not sample_from_test:
                    target_indices_batch = np.where(labels.cpu().numpy() == target_class)[0]
                    available_indices = list(set(target_indices_batch) - set(pos_indices))
                    if len(available_indices) < len(pos_indices):
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices = available_indices 
                    else:
                        extra_clean_indices = []
                        clean_indices = random.sample(available_indices, len(pos_indices)) 
                    available_indices = list(set(target_indices_batch) - set(pos_indices) - set(clean_indices))
                    if len(available_indices) < len(pos_indices) and len(available_indices) != 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices) - len(available_indices))
                        clean_indices_2 = available_indices
                    elif len(available_indices) == 0:
                        indices_to_ignore = set(random_sus_idx) | set(indices[available_indices].cpu().numpy())
                        remaining_target_indices = [i for i, idx in enumerate(target_indices) if idx not in indices_to_ignore]
                        extra_clean_indices_2 = random.sample(remaining_target_indices, len(pos_indices))
                        clean_indices_2 = []
                    else:
                        extra_clean_indices_2 = []
                        clean_indices_2 = random.sample(available_indices, len(pos_indices))
                    

                    assert set(indices[pos_indices].cpu().numpy()) & set(random_sus_idx) == set(indices[pos_indices].cpu().numpy())
                    assert np.all(labels[clean_indices].cpu().numpy() == target_class)
                    assert np.all(labels[clean_indices_2].cpu().numpy() == target_class)
                    
                else:
                    clean_indices = list(random.sample(range(len(target_images)), len(pos_indices)))
                    clean_indices_2 = list(random.sample(set(range(len(target_images))) - set(clean_indices), len(pos_indices)))
                    
                    assert set(clean_indices) & set(clean_indices_2) == set()
                    assert len(clean_indices) == len(clean_indices_2) == len(pos_indices)
              
                
                step3_time = time.time() - start_time
                start_time = time.time()
                step3_time_avg += step3_time
                
                
                
                combined_batch = []
                combined_labels = []
                combined_indexes = []
                pos_indexes = indices[pos_indices].cpu().numpy()
                
                
                # Add suspected samples
                combined_batch.extend(images[pos_indices])
                combined_labels.extend(labels[pos_indices])
                
                
                if not sample_from_test: 
                    if len(extra_clean_indices) > 0:
                        clean_batch = torch.cat([images[clean_indices], target_images[extra_clean_indices]])
                        clean_labels = torch.cat([labels[clean_indices], torch.tensor([target_class] * len(extra_clean_indices)).to(device)])
                        clean_indexes = np.concatenate([indices[clean_indices].cpu().numpy(), np.array(target_indices)[extra_clean_indices]])
                    else:
                        clean_batch = images[clean_indices]
                        clean_labels = labels[clean_indices]
                        clean_indexes = indices[clean_indices].cpu().numpy()
                else:
                    clean_batch = target_images[clean_indices]
                    clean_labels = torch.tensor([target_class] * len(clean_indices)).to(device)
                    clean_indexes = np.array(clean_indices) + 50000
                
                step_4_1_time = time.time() - start_time
                step_4_1_time_avg += step_4_1_time
                start_time = time.time()

                # Add clean_1 samples
                combined_batch.extend(clean_batch)
                combined_labels.extend(clean_labels)
                
                
                if not sample_from_test: 
                    if len(extra_clean_indices_2) > 0:
                        clean_batch_2 = torch.cat([images[clean_indices_2], target_images[extra_clean_indices_2]])
                        clean_labels_2 = torch.cat([labels[clean_indices_2 ], torch.tensor([target_class] * len(extra_clean_indices_2)).to(device)])
                        clean_indexes_2 = np.concatenate([indices[clean_indices_2].cpu().numpy(), np.array(target_indices)[extra_clean_indices_2]])
                    else:
                        clean_batch_2 = images[clean_indices_2]
                        clean_labels_2 = labels[clean_indices_2]
                        clean_indexes_2 = indices[clean_indices_2].cpu().numpy()
                else:
                    clean_batch_2 = target_images[clean_indices_2]
                    clean_labels_2 = torch.tensor([target_class] * len(clean_indices_2)).to(device)
                    clean_indexes_2 = np.array(clean_indices_2) + 50000
                
                
                tagged_clean_indexes = [(index, 1) for index in clean_indexes] 
                tagged_clean_indexes_2 = [(index, 2) for index in clean_indexes_2] 
                tagged_pos_indexes = [(index, 0) for index in pos_indexes] 

                combined_indexes = tagged_pos_indexes + tagged_clean_indexes + tagged_clean_indexes_2
                
                
                step_4_2_time = time.time() - start_time
                step_4_2_time_avg += step_4_2_time
                start_time = time.time()


                combined_batch.extend(clean_batch_2)
                combined_labels.extend(clean_labels_2)

                combined_batch = torch.stack(combined_batch).to(device)
                combined_labels = torch.tensor(combined_labels).to(device)
     
                gen = torch.Generator().manual_seed(seed)

                combined_loader = DataLoader(
                    list(zip(combined_batch, combined_labels, combined_indexes)),
                    batch_size=1,
                    shuffle=False,  
                    generator=gen 
                )

                temp_sus = {}
                temp_clean = {}
                temp_clean_2 = np.zeros(len(important_features))  
                batch_count_clean_2 = 0

                
                for image, label, (index, tag) in combined_loader:
                    torch_rng_state = torch.get_rng_state()
                    cuda_rng_state = torch.cuda.get_rng_state()
                    np_rng_state = np.random.get_state()
                    python_rng_state = random.getstate()
    
                    sur_model.load_state_dict(original_weights)
                    sur_optimizer.load_state_dict(optimizer_state)
                    sur_model.train(mode=training_mode)
                    sur_optimizer.zero_grad()

                    # try:
                    output = sur_model(image)
                    # except ValueError as e:
                    #     # upsample the image to avoid batchnorm error
                    #     image_rep = image.repeat(2, 1, 1, 1) 
                    #     output = sur_model(image_rep)[0].unsqueeze(0)
                    
                    step_4_3_time = time.time() - start_time
                    step_4_3_time_avg += step_4_3_time
                    start_time = time.time()

                    clean_label = label.long()
                    loss = criterion(output, clean_label)
                    loss.backward()
                    sur_optimizer.step()

                    sur_model_state_dict = sur_model.state_dict()
                    
                    step_4_4_time = time.time() - start_time
                    step_4_4_time_avg += step_4_4_time
                    start_time = time.time()
            
                    
                    flat_sur_weights = torch.cat([param.flatten() for  param in sur_model_state_dict.values()])
                    flat_orig_weights = torch.cat([param.flatten() for  param in original_weights.values()])
                    important_flat_indices = torch.tensor(important_features).to(device)
                    important_diff = flat_sur_weights[important_flat_indices] - flat_orig_weights[important_flat_indices]

                    increment = important_diff.cpu().numpy()

                    
                    step_4_5_time = time.time() - start_time
                    step_4_5_time_avg += step_4_5_time
                    start_time = time.time()

                    if tag == 0:  # Suspected samples
                        temp_sus[index] = increment
                    elif tag == 1:  # Clean_1 samples
                        temp_clean[index] = increment
                    elif tag == 2:  # Clean_2 samples
                        batch_count_clean_2 += 1
                        temp_clean_2 = temp_clean_2 + (increment - temp_clean_2) / batch_count_clean_2
                        
                    step5_time = time.time() - start_time
                    step5_time_avg += step5_time
                    start_time = time.time()
                    
                    torch.set_rng_state(torch_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state)
                    np.random.set_state(np_rng_state)
                    random.setstate(python_rng_state)
    

                # Calculate sus_diff and clean_diff
                for index in temp_sus:
                    sus_diff[(epoch, index)] = temp_sus[index] - temp_clean_2
                    
                    
                step6_time = time.time() - start_time
                step6_time_avg += step6_time
                start_time = time.time()

                for index in temp_clean:
                    clean_diff[(epoch, index)] = temp_clean[index] - temp_clean_2

                
                    
                
                step7_time = time.time() - start_time
                step7_time_avg += step7_time
                step7_time_avg += step7_time
                
                del temp_sus, temp_clean, temp_clean_2
                
        
                

        torch.cuda.empty_cache()       
        
        
        train_ACC.append(acc_meter.avg)
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

        
        
        start_time = time.time()
        
        
        
        step7_time = time.time() - start_time
        step7_time_avg += step7_time
        
        
        
        # Testing 
        if type(poisoned_test_loader) == dict:
            for attack_name in poisoned_test_loader:
                print(f"Testing attack effect for {attack_name}")
                model.eval()
                correct, total = 0, 0
                for i, (images, labels) in enumerate(poisoned_test_loader[attack_name]):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        logits = model(images)
                        out_loss = criterion(logits,labels)
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                acc = correct / total
                test_ACC.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            model.eval()
            correct, total = 0, 0
            for i, (images, labels) in enumerate(poisoned_test_loader):
                images, labels = images.to(device).float(), labels.to(device).long()
                with torch.no_grad():
                    logits = model(images)
                    out_loss = criterion(logits,labels)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            test_ACC.append(acc)
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
        
        correct_clean, total_clean = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device).float(), labels.to(device)
            with torch.no_grad():
                logits = model(images)
                out_loss = criterion(logits,labels)
                _, predicted = torch.max(logits.data, 1)
                total_clean += labels.size(0)
                correct_clean += (predicted == labels).sum().item()
        acc_clean = correct_clean / total_clean
        clean_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:', out_loss)
    
    
    sus_inds = np.array([ind.item() for epoch, ind in sus_diff])
    clean_inds = np.array([ind.item() for epoch, ind in clean_diff])
    sus_diff = np.array([sus_diff[key] for key in sus_diff])
    clean_diff = np.array([clean_diff[key] for key in clean_diff])
    
    return sus_diff, clean_diff, sus_inds, clean_inds
