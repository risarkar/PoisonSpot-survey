import torch
from tqdm import tqdm
import numpy as np
import random
from src.utils.util import AverageMeter

__all__ = ["train", "evaluate_model"]

def train(
    model,
    optimizer,
    opt,
    scheduler,
    criterion,
    poisoned_train_loader,
    test_loader,
    poisoned_test_loader,
    training_epochs,
    global_seed,
    device,
    training_mode=True
):
    """
    Train and evaluate the model on clean and poisoned data for a given number of epochs.

    Args:
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        opt (dict): Dictionary of optimizer hyperparameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (callable): Loss function.
        poisoned_train_loader (DataLoader): DataLoader for the poisoned training set.
        test_loader (DataLoader): DataLoader for the clean test set.
        poisoned_test_loader (DataLoader): DataLoader for the poisoned test set.
        training_epochs (int): Number of training epochs.
        global_seed (int): Random seed for reproducibility.
        device (torch.device): Device to run training on.
        training_mode (bool, optional): If True, run training; otherwise only evaluate. Defaults to True.

    Returns:
        dict: Metrics recorded per epoch (training loss/accuracy, test accuracy, backdoor ASR).
    """    
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)
    
    test_ASR = []
    test_ACC = []
    model.to(device)
    
    for _ in tqdm(range(training_epochs)):
        # Train
        model.train(mode = training_mode)
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm(poisoned_train_loader, total=len(poisoned_train_loader)) 
        for images, labels, indices in pbar:
            images, labels, indices = images.to(device).float() , labels.to(device).long(), indices            
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
        print('Train_loss:',loss)
        if opt == 'sgd':
            scheduler.step()

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
                test_ASR.append(acc)
                print('\nAttack success rate %.2f' % (acc*100))
                print('Test_loss:',out_loss)
        else:
            # Testing attack effect
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
            test_ASR.append(acc)
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
        test_ACC.append(acc_clean)
        print('\nTest clean Accuracy %.2f' % (acc_clean*100))
        print('Test_loss:', out_loss)
    
        
    return model, optimizer, scheduler, test_ASR, test_ACC



def evaluate_model(model, test_loader, poisoned_test_loader, criterion, device):
    model.eval()
    model.to(device)
    
    
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
            print('\nAttack success rate %.2f' % (acc*100))
            print('Test_loss:',out_loss)
    else:
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
    print('\nTest clean Accuracy %.2f' % (acc_clean*100))
    print('Test_loss:',out_loss)
    