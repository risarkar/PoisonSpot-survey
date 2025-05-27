import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

__all__ = ["get_loaders_from_dataset", "get_random_poison_idx"]

def get_loaders_from_dataset(
    poisoned_train_dataset,
    test_dataset, 
    poisoned_test_dataset, 
    batch_size, target_class, 
    indexes_to_remove = []
    ):
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    if type(poisoned_test_dataset) == dict:
        poisoned_test_loader = {}
        for attack_name in poisoned_test_dataset:
            poisoned_test_loader[attack_name] = DataLoader(poisoned_test_dataset[attack_name], batch_size=batch_size, shuffle=False, num_workers=2)
    else:     
        poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if len(indexes_to_remove) > 0:
        print("Length of indexes to remove: ", len(indexes_to_remove))
        
        indices = [idx.item() if isinstance(idx, torch.Tensor) else idx for img, lbl, idx in poisoned_train_dataset]

        print(len(set(indices) & set(indexes_to_remove)))
        indexes_to_keep = [i for i, idx in enumerate(indices) if idx not in indexes_to_remove]
        poisoned_train_dataset_filtered = Subset(poisoned_train_dataset, indexes_to_keep)
        target_class_indices = [idx for img, lbl, idx in poisoned_train_dataset_filtered if lbl == target_class]
        
        print("Length of the filtered dataset: ", len(poisoned_train_dataset_filtered))
        poisoned_train_loader = DataLoader(poisoned_train_dataset_filtered, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        if type(poisoned_train_dataset[0][2]) == torch.Tensor:
            target_class_indices = [idx.item() for img, lbl, idx in poisoned_train_dataset if lbl == target_class]
        else:
            target_class_indices = [idx for img, lbl, idx in poisoned_train_dataset if lbl == target_class]
    
    
    
    return poisoned_train_loader, test_loader, poisoned_test_loader, target_class_indices
    


def get_random_poison_idx(
    percentage, ignore_set, random_poison_idx,
    target_class_all, poison_amount, seed,
):
    """
    Randomly pick additional *clean* indices from the target class so that the
    suspected set has   (#poison / percentage)  total size.
    """
    np.random.seed(seed)
    random.seed(seed)
    extra = int(poison_amount * (100 / percentage - 1))
    clean_pool = list(set(target_class_all) - ignore_set)
    return random.sample(clean_pool, extra) + list(random_poison_idx)

    