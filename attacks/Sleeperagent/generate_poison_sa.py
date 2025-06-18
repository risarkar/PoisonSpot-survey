import numpy as np
import os
from PIL import Image
from numpy import asarray
from skimage.transform import resize
import random
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets


import sys
from src.models.resnet import ResNet
from art.utils import to_categorical
from art.attacks.poisoning.sleeper_agent_attack import SleeperAgentAttack
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, ToPILImage, Resize
from pytorch_pretrained_vit import ViT

deterministic = True

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def get_sa_cifar10_poisoned_data(
    poison_ratio=10,
    target_class=1,
    source_class=0,
    datasets_root_dir='./src/data/',
    model=ResNet(18),
    clean_model_path='./src/saved_models/',
    global_seed=545,
    random_sa=False,
    gpu_id=0,
    optimizer=None
):
    """
    Generate and return CIFAR-10 poisoned dataset using Sleeper Agent attack.

    Arguments:
        poison_ratio (float): percentage of poisoned samples in target class.
        target_class (int): Label to be targeted for poisoning.
        source_class (int): backdoor trigger is applied to this class at test time to be misclassified as target_class.
        model (torch.nn.Module): Predefined PyTorch model architecture.
        dataset_path (str): Directory to save or load the poisoned data.
        clean_model_path (str): File path to save or load the clean pretrained model. 
        global_seed (int): Seed for reproducibility 
        gpu_id (int): GPU identifier for CUDA environment variable.

    Returns:
        poisoned_train_dataset (TensorDataset): Poisoned training set with indices.
        test_dataset (TensorDataset): Clean test set.
        poisoned_test_dataset (TensorDataset): Test set with poisoned samples.
        poison_indices (ndarray): Indices of samples poisoned in training set.
    """  
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)

    poison_ratio = poison_ratio / 100.0
    (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))

    patch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open('./src/attacks/Sleeperagent/trigger_10.png')
    numpydata = asarray(img)
    patch = resize(numpydata, (patch_size,patch_size,3))
    patch = np.transpose(patch,(2,0,1))
    x_train_orig = np.copy(x_train)
    K = 1000
    


    def select_trigger_train(x_train,y_train,K,source_class,target_class):
        x_train_ = np.copy(x_train)
        index_source = np.where(y_train.argmax(axis=1)==source_class)[0][0:K]
        index_target = np.where(y_train.argmax(axis=1)==target_class)[0]
        x_trigger = x_train_[index_source]
        y_trigger  = to_categorical([target_class], nb_classes=10)
        y_trigger = np.tile(y_trigger,(len(index_source),1))
        return x_trigger,y_trigger,index_target
    
    
    def add_trigger_patch(x_set,patch_type="fixed"):
        img = Image.open('./src/attacks/Sleeperagent/trigger_10.png')
        numpydata = asarray(img)
        patch = resize(numpydata, (patch_size,patch_size,3))
        patch = np.transpose(patch,(2,0,1))
        if patch_type == "fixed":
            x_set[:,:,-patch_size:,-patch_size:] = patch
        else:
            for x in x_set:
                x_cord = random.randrange(0,x.shape[1] - patch.shape[1] + 1)
                y_cord = random.randrange(0,x.shape[2] - patch.shape[2] + 1)
                x[:,x_cord:x_cord+patch_size,y_cord:y_cord+patch_size]=patch

        return x_set

    if not random_sa:
        indices_path = datasets_root_dir + f'indices_poison_resnet18_sa_{target_class}_{source_class}_16_{poison_ratio}_128.npy'
        x_poison_path = datasets_root_dir + f'x_poison_resnet18_sa_{target_class}_{source_class}_16_{poison_ratio}_128.npy'
        y_poison_path = datasets_root_dir + f'y_poison_resnet18_sa_{target_class}_{source_class}_16_{poison_ratio}_128.npy'
    else:
        indices_path = datasets_root_dir + f'indices_poison_resnet_custom_sa_{target_class}_{source_class}_16_{poison_ratio}.npy'
        x_poison_path = datasets_root_dir + f'x_poison_resnet_custom_sa_{target_class}_{source_class}_16_{poison_ratio}.npy'
        y_poison_path = datasets_root_dir + f'y_poison_resnet_custom_sa_{target_class}_{source_class}_16_{poison_ratio}.npy'
    
    if not os.path.exists(indices_path) or not os.path.exists(x_poison_path) or not os.path.exists(y_poison_path):
        print("Generating the attack")
        loss_fn = nn.CrossEntropyLoss()
        
        model_art = PyTorchClassifier(model,input_shape=x_train.shape[1:], loss=loss_fn, optimizer=optimizer, nb_classes=10, clip_values=(min_, max_), preprocessing=(mean,std))
        model.load_state_dict(torch.load(clean_model_path))
        x_trigger,y_trigger,index_target = select_trigger_train(x_train,y_train,K,source_class,target_class)
        attack = SleeperAgentAttack(model_art,
                                        percent_poison= poison_ratio,
                                        max_trials=4,
                                        max_epochs=500,
                                        learning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]), [250, 350, 400, 430, 460]),
                                        epsilon=32/255,
                                        batch_size=500,
                                        verbose=1,
                                        indices_target=index_target,
                                        patching_strategy="fixed",
                                        selection_strategy="max-norm",
                                        patch=patch,
                                        retraining_factor = 4,
                                        model_retrain = True,
                                        model_retraining_epoch = 80,
                                        retrain_batch_size = 128,
                                        source_class = source_class,
                                        target_class = target_class,
                                        device_name = str(device)       
                                )
        x_poison, y_poison = attack.poison(x_trigger,y_trigger,x_train,y_train,x_test,y_test) 
        indices_poison = attack.get_poison_indices()
        
        np.save(x_poison_path,x_poison)
        np.save(y_poison_path,y_poison)
        np.save(indices_path,indices_poison)
    
    x_poison = np.load(x_poison_path)
    y_poison = np.load(y_poison_path)
    poison_indices = np.load(indices_path) 
    
    all_indices = np.arange(len(x_poison))
    poisoned_train_dataset = TensorDataset(torch.tensor(x_poison), torch.tensor(y_poison.argmax(axis=1)), torch.tensor(all_indices))
    test_dataset = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test.argmax(axis=1)).long())
    
    index_source_test = np.where(y_test.argmax(axis=1)==source_class)[0]
    x_test_trigger = x_test[index_source_test]
    if not random_sa:
        x_test_trigger = add_trigger_patch(x_test_trigger,"fixed")
    else:
        x_test_trigger = add_trigger_patch(x_test_trigger,"random")
    y_test_trigger = np.ones(len(x_test_trigger))*target_class


    index_source_train = np.where(y_poison.argmax(axis=1)==target_class)[0]
    poison_indices = index_source_train[poison_indices]
    
    poisoned_test_dataset = TensorDataset(torch.tensor(x_test_trigger).float(), torch.tensor(y_test_trigger).long())
    
    
    class TransformedTensorDataset(Dataset):
        def __init__(self, tensor_dataset, transform=None):
            self.tensor_dataset = tensor_dataset
            self.transform = transform
            self.to_pil = ToPILImage()

        def __len__(self):
            return len(self.tensor_dataset)

        def __getitem__(self, idx):
            data = self.tensor_dataset[idx]
            try:
                image, label = data
                image = self.to_pil(image)
                if self.transform:
                    image = self.transform(image)
                return image, label
            except:
                image, label, index = data
                image = self.to_pil(image)
                if self.transform:
                    image = self.transform(image)
                return image, label, index
    

    transform_train = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # RandomHorizontalFlip(),
    ])
    
    transform_test = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    poisoned_train_dataset = TransformedTensorDataset(poisoned_train_dataset, transform=transform_train)
    test_dataset = TransformedTensorDataset(test_dataset, transform=transform_test)
    poisoned_test_dataset = TransformedTensorDataset(poisoned_test_dataset, transform=transform_test)
    
    return poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices 
    
    
  
def get_sa_slt_10_poisoned_data(
    poison_ratio=10, 
    target_class = 1, 
    source_class = 0, 
    datasets_root_dir='./src/data/', 
    model = ResNet(18), 
    clean_model_path = './src/saved_models/',
    global_seed=545, 
    gpu_id=0, 
    optimizer = None
    ):
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    
    poison_ratio = poison_ratio / 100.0
    
    transform = Compose([
        Resize((224, 224)), 
        ToTensor(),  
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # Load the STL-10 dataset
    train_dataset = datasets.STL10(root=datasets_root_dir, split='train', transform=transform, download=False)
    test_dataset = datasets.STL10(root=datasets_root_dir, split='test', transform=transform, download=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Collect transformed data
    def collect_transformed_data(loader):
        transformed_data = []
        transformed_labels = []
        for data, labels in loader:
            transformed_data.append(data)
            transformed_labels.append(labels)
        return torch.cat(transformed_data), torch.cat(transformed_labels)

    x_test, y_test = collect_transformed_data(test_loader)
    x_test, y_test = x_test.numpy(), y_test.numpy()
    y_test = np.eye(10)[y_test]


    patch_size = 16
    img = Image.open('./src/attacks/Sleeperagent/trigger_10.png')
    numpydata = asarray(img)
    patch = resize(numpydata, (patch_size,patch_size,3))
    patch = np.transpose(patch,(2,0,1))
    

    
    
    def add_trigger_patch(x_set,patch_type="fixed"):
        img = Image.open('./src/attacks/Sleeperagent/trigger_10.png')
        numpydata = asarray(img)
        patch = resize(numpydata, (patch_size,patch_size,3))
        patch = np.transpose(patch,(2,0,1))
        if patch_type == "fixed":
            x_set[:,:,-patch_size:,-patch_size:] = patch
        else:
            for x in x_set:
                x_cord = random.randrange(0,x.shape[1] - patch.shape[1] + 1)
                y_cord = random.randrange(0,x.shape[2] - patch.shape[2] + 1)
                x[:,x_cord:x_cord+patch_size,y_cord:y_cord+patch_size]=patch

        return x_set
    
            
    print("poison_ratio: ", poison_ratio)
    indices_path = datasets_root_dir + f'indices_poison_sa_vit_{target_class}_{source_class}_16_{poison_ratio}_32.npy'
    x_poison_path = datasets_root_dir + f'x_poison_sa_vit_{target_class}_{source_class}_16_{poison_ratio}_32.npy'
    y_poison_path = datasets_root_dir + f'y_poison_sa_vit_{target_class}_{source_class}_16_{poison_ratio}_32.npy'
    
    x_poison = np.load(x_poison_path)
    y_poison = np.load(y_poison_path)
    poison_indices = np.load(indices_path) 
    
    index_source_train = np.where(y_poison.argmax(axis=1)==target_class)[0]
    poison_indices = index_source_train[poison_indices]
    
    all_indices = np.arange(len(x_poison))
    poisoned_train_dataset = TensorDataset(torch.tensor(x_poison), torch.tensor(y_poison.argmax(axis=1)), torch.tensor(all_indices))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.argmax(axis=1)))
    index_source_test = np.where(y_test.argmax(axis=1)==source_class)[0]
    x_test_trigger = x_test[index_source_test]
    x_test_trigger = add_trigger_patch(x_test_trigger,"fixed")
    y_test_trigger = np.ones(len(x_test_trigger))*target_class

    poisoned_test_dataset = TensorDataset(torch.tensor(x_test_trigger), torch.tensor(y_test_trigger))
    
    return poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices 
    
    
    
    