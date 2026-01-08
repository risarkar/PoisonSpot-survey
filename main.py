
"""
PoisonSpot: Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking
"""

# from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import time
import copy
import random
import warnings
import pickle
from types import SimpleNamespace

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")            

import torch
import torch.nn as nn
import torchvision
import logging

from src import *
import csv
import json
import shutil
warnings.filterwarnings("ignore")

import tensorflow as tf                                        
tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel("ERROR")


def parse_args() -> argparse.Namespace:
    """CLI wrapper – only a YAML config path is required."""
    ap = argparse.ArgumentParser(description="PoisonSpot Defence")
    ap.add_argument(
        "-c", "--config",
        default="configs/config_lc_cifar_10.yaml",
        help="Path to the YAML configuration file",
    )
    return ap.parse_args()


def load_cfg(path: str) -> SimpleNamespace:
    """Load a YAML file and make its keys dot-accessible."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    def _to_ns(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = _to_ns(v)
        return SimpleNamespace(**d)

    return _to_ns(data)


def main() -> None:
    print(torch.cuda.device_count(), "GPUs available")
    args = parse_args()
    cfg = load_cfg(args.config)         
    
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    if cfg.exp is None:
        cfg.exp  = f"{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.pr_sus}"

    # seed for reproducibility
    torch.manual_seed(cfg.global_seed)
    np.random.seed(cfg.global_seed)
    random.seed(cfg.global_seed)

    # summary heade
    print(
        f"Attack: {cfg.attack} | Dataset: {cfg.dataset} | Model: {cfg.model} | "
        f"Target-cls: {cfg.target_class} | Poison-ratio: {cfg.pr_tgt} | "
        f"Suspected %: {cfg.pr_sus}"
    )

    device = torch.device(f"cuda:{cfg.gpu_id}")
    print(f"Using GPU {cfg.gpu_id}")

    # models
    if cfg.model == "ResNet18":
        orig_model = ResNet(18)
    elif cfg.model == "CustomCNN":
        orig_model = CustomCNN()
    elif cfg.model == "BasicResNet":
        orig_model = torchvision.models.ResNet(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=10
        )
    elif cfg.model == "CustomResNet18":
        orig_model = CustomResNet18()
    elif cfg.model == "ViT" and cfg.scenario == "fine_tuning":
        orig_model = CustomViT("B_16_imagenet1k", pretrained=True)
        orig_model.fc = nn.Linear(orig_model.fc.in_features,
                                  10 if cfg.dataset in {"slt10", "CIFAR10"} else 100)
    elif cfg.model == "ViT" and cfg.scenario == "from_scratch":
        raise ValueError("ViT from-scratch not supported")
    else:
        raise ValueError(f"Unknown model {cfg.model}")

    if (cfg.scenario == "fine_tuning"
        and (cfg.attack, cfg.dataset) not in {("sa", "slt10"), ("lc", "imagenet")}):
        orig_model.load_state_dict(torch.load(cfg.clean_model_path))

        if cfg.attack == "ht":
            for p in orig_model.parameters():
                p.requires_grad = False
            if cfg.dataset == "CIFAR10":
                orig_model[20] = nn.Linear(4096, 10)
            elif cfg.dataset == "slt10":
                for p in orig_model.fc.parameters():
                    p.requires_grad = True
            elif cfg.dataset == "imagenet":
                for p in list(orig_model.fc1.parameters()) + list(orig_model.fc2.parameters()):
                    p.requires_grad = True

    elif (cfg.attack, cfg.dataset) in {("sa", "slt10"), ("lc", "imagenet")}:
        orig_model = orig_model.to(device)
    elif cfg.scenario != "from_scratch":
        raise ValueError("Unsupported scenario")

    if cfg.random:
        cfg.exp += "_random"
        cfg.saved_models_path += "random/"
        cfg.prov_path += "random/"
    if not os.path.exists(cfg.saved_models_path):
        os.makedirs(cfg.saved_models_path)
    if not os.path.exists(cfg.prov_path):
        os.makedirs(cfg.prov_path)

        
    if cfg.pr_sus == int(cfg.pr_sus):
        cfg.pr_sus = int(cfg.pr_sus)

    # Load the dataset 
    if cfg.dataset == "CIFAR10":
        if cfg.attack == "lc":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_lc_cifar10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.dataset_dir,
                    copy.deepcopy(orig_model),
                    cfg.clean_model_path,
                    cfg.eps,
                    cfg.vis,
                    cfg.global_seed,
                    cfg.gpu_id,
                )
        elif cfg.attack == "narcissus":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_narcissus_cifar10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.dataset_dir,
                    copy.deepcopy(orig_model),
                    cfg.eps,
                    cfg.global_seed,
                )
        elif cfg.attack == "sa":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_sa_cifar10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.source_class,
                    cfg.dataset_dir,
                    copy.deepcopy(orig_model),
                    cfg.clean_model_path,
                    global_seed=cfg.global_seed,
                    random_sa=cfg.random,
                )
        elif cfg.attack == "ht":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_ht_cifar10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.source_class,
                    copy.deepcopy(orig_model),
                    cfg.dataset_dir,
                    cfg.clean_model_path,
                    cfg.global_seed,
                )
        elif cfg.attack == "narcissus_lc":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices_all = \
                get_lc_narcissus_cifar_10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.dataset_dir,
                    copy.deepcopy(orig_model),
                    cfg.clean_model_path,
                    cfg.vis,
                    cfg.global_seed,
                    cfg.gpu_id,
                )
            poison_indices = np.concatenate(list(poison_indices_all.values()))
        elif cfg.attack == "narcissus_lc_sa":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices_all = \
                get_lc_narcissus_sa_cifar_10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.dataset_dir,
                    copy.deepcopy(orig_model),
                    cfg.clean_model_path,
                    cfg.vis,
                    cfg.global_seed,
                    cfg.gpu_id,
                )
            poison_indices = np.concatenate(list(poison_indices_all.values()))
        else:
            raise ValueError(f"Attack {cfg.attack} not supported on CIFAR10")

    elif cfg.dataset == "slt10":
        if cfg.attack == "sa":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_sa_slt_10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.source_class,
                    cfg.dataset_dir,
                    copy.deepcopy(orig_model),
                    cfg.clean_model_path,
                    cfg.global_seed,
                )
        elif cfg.attack == "ht":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_ht_stl10_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.source_class,
                    copy.deepcopy(orig_model),
                    cfg.dataset_dir,
                    cfg.clean_model_path,
                    cfg.global_seed,
                )
        else:
            raise ValueError(f"Attack {cfg.attack} not supported on SLT10")

    elif cfg.dataset == "imagenet":
        if cfg.attack == "ht":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_ht_imagenet_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.source_class,
                    copy.deepcopy(orig_model),
                    cfg.dataset_dir,
                    cfg.clean_model_path,
                    cfg.global_seed,
                )
        elif cfg.attack == "lc":
            poisoned_train_dataset, test_dataset, poisoned_test_dataset, poison_indices = \
                get_lc_image_net_poisoned_data(
                    cfg.pr_tgt,
                    cfg.target_class,
                    cfg.dataset_dir,
                    copy.deepcopy(orig_model),
                    cfg.clean_model_path,
                    cfg.eps,
                    cfg.global_seed,
                    cfg.gpu_id,
                )
        else:
            raise ValueError(f"Attack {cfg.attack} not supported on ImageNet")

    else:
        raise ValueError(f"Dataset {cfg.dataset} not recognized")  
    

    # Initialize the experiment folder and CSV file
    def init_experiment_folder(cfg):
        """Create (or locate) the experiment folder and open the CSV once."""
        folder = os.path.join(cfg.results_path, cfg.exp)
        os.makedirs(folder, exist_ok=True)


        cfg_path = os.path.join(folder, "config.json")
        with open(cfg_path, "w") as j:
            json.dump(vars(cfg), j, indent=4)

        header = [
            "epochs",
            "Clean training ACC",
            "Poisoned training ACC", "Poisoned training ASR",
            "Batch-level |features|",
            "TPR KMeans", "FPR KMeans",
            "TPR Gaussian", "FPR Gaussian",
            "Retrain ACC", "Retrain ASR"
        ]
        csv_path = os.path.join(folder, "results.csv")

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

        return csv_path
    RESULTS_HEADER = [
        "epochs",
        "Clean training ACC",
        "Poisoned training ACC", "Poisoned training ASR",
        "Batch-level |features|",
        "TPR KMeans", "FPR KMeans",
        "TPR Gaussian", "FPR Gaussian",
        "Retrain ACC", "Retrain ASR"
    ]

    def update_results(csv_path, **metrics):
        """
        Append a row with the supplied metrics.
        """
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            header = list(metrics.keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerow(metrics)
            return

        # Read existing CSV
        with open(csv_path, newline="") as f:
            reader = list(csv.DictReader(f))
            old_fieldnames = reader[0].keys() if reader else []
            rows = [dict(r) for r in reader]

        # Determine new header
        new_keys = [k for k in metrics.keys() if k not in old_fieldnames]
        fieldnames = list(old_fieldnames) + new_keys

        # Rewrite CSV with updated header
        tmp_path = csv_path + ".tmp"
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # write old rows 
            for r in rows:
                writer.writerow(r)
            # write the new row
            writer.writerow({**{k: "" for k in fieldnames}, **metrics})

        # Replace original file
        shutil.move(tmp_path, csv_path)

    if cfg.clean_training:
        logger.info("Starting clean training...")
        train_loader, test_loader, poisoned_test_loader, _ = get_loaders_from_dataset(
            poisoned_train_dataset,
            test_dataset,
            poisoned_test_dataset,
            cfg.bs,
            cfg.target_class,
            poison_indices,
        )

        model = copy.deepcopy(orig_model).to(device)

        # pick optimizer 
        if cfg.opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        elif cfg.opt == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=5e-4,
                dampening=0,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.opt}")

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        if cfg.get_result:

            ckpt = (
                f"{cfg.saved_models_path}"
                f"clean_model_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.epochs}.pkl"
            )
            model.load_state_dict(torch.load(ckpt))
            test_ASR, test_ACC = evaluate_model(model, test_loader, poisoned_test_loader, criterion, device)
            
            csv_file = init_experiment_folder(cfg)   
            update_results(
                csv_file,
                epochs=cfg.epochs,
                **{
                    "Clean training ACC": test_ACC,
                }
            )
        else:
            # train & save
            model, optimizer, scheduler, test_ASR, test_ACC = train(
                model,
                optimizer,
                cfg.opt,
                scheduler,
                criterion,
                train_loader,
                test_loader,
                poisoned_test_loader,
                cfg.epochs,
                cfg.global_seed,
                device,
                cfg.training_mode,
            )
            ckpt = (
                f"{cfg.saved_models_path}"
                f"clean_model_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.epochs}.pkl"
            )
            torch.save(model.state_dict(), ckpt)
            
            csv_file = init_experiment_folder(cfg)   
            update_results(
                csv_file,
                epochs=cfg.epochs,
                **{
                    "Clean training ACC": test_ACC[-1],
                }
            )

                
            

        
    if cfg.poisoned_training:
        logger.info("Starting poisoned training...")
        poisoned_train_loader, test_loader, poisoned_test_loader, _ = get_loaders_from_dataset(
            poisoned_train_dataset,
            test_dataset,
            poisoned_test_dataset,
            cfg.bs,
            cfg.target_class,
        )
        model = copy.deepcopy(orig_model).to(device)

        if cfg.opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        elif cfg.opt == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=5e-4,
                dampening=0,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.opt}")

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        ckpt = (
            f"{cfg.saved_models_path}"
            f"model_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.epochs}_{cfg.bs}.pkl"
        )

        if cfg.get_result:
            # Load and evaluate
            model.load_state_dict(torch.load(ckpt))
            test_ASR, test_ACC = evaluate_model(model, test_loader, poisoned_test_loader, criterion, device)
            
            if cfg.attack in {"narcissus_lc"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Poisoned training ASR Narcissus": test_ASR[-1],
                        "Poisoned training ASR Label Consistent": test_ASR[-2],
                        "Poisoned training ACC": test_ACC,
                    }
                )
            elif cfg.attack in {"narcissus_lc_sa"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Poisoned training ASR Narcissus": test_ASR[-1],
                        "Poisoned training ASR Label Consistent": test_ASR[-2],
                        "Poisoned training ASR Sleeper Agent": test_ASR[-3],
                        "Poisoned training ACC": test_ACC,
                    }
                )
            else:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Poisoned training ACC": test_ACC,
                        "Poisoned training ASR": test_ASR[-1],
                    }
                )
            
        else:
            # Train and save
            model, optimizer, scheduler, test_ASR, test_ACC= train(
                model,
                optimizer,
                cfg.opt,
                scheduler,
                criterion,
                poisoned_train_loader,
                test_loader,
                poisoned_test_loader,
                cfg.epochs,
                cfg.global_seed,
                device,
                cfg.training_mode,
            )
            torch.save(model.state_dict(), ckpt)
            torch.save(
                optimizer.state_dict(),
                f"{cfg.saved_models_path}"
                f"optimizer_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.epochs}_{cfg.bs}.pkl",
            )
            torch.save(
                scheduler.state_dict(),
                f"{cfg.saved_models_path}"
                f"scheduler_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.epochs}_{cfg.bs}.pkl",
            )

            if cfg.attack in {"narcissus_lc"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Poisoned training ASR Narcissus": test_ASR[-1],
                        "Poisoned training ASR Label Consistent": test_ASR[-2],
                        "Poisoned training ACC": test_ACC,
                    }
                )
            elif cfg.attack in {"narcissus_lc_sa"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Poisoned training ASR Narcissus": test_ASR[-1],
                        "Poisoned training ASR Label Consistent": test_ASR[-2],
                        "Poisoned training ASR Sleeper Agent": test_ASR[-3],
                        "Poisoned training ACC": test_ACC,
                    }
                )
            else:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Poisoned training ACC": test_ACC[-1],
                        "Poisoned training ASR": test_ASR[-1],
                    }
                )
        
    if cfg.batch_level:
        logger.info("Starting batch‐level provenance capture...")
        poisoned_train_loader, test_loader, poisoned_test_loader, target_class_indices = get_loaders_from_dataset(
            poisoned_train_dataset,
            test_dataset,
            poisoned_test_dataset,
            cfg.bs_bl,
            cfg.target_class
        )
        model = copy.deepcopy(orig_model).to(device)

        # Set up optimizer
        if cfg.opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        elif cfg.opt == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=5e-4,
                dampening=0,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.opt}")

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.ep_bl_base + cfg.ep_bl
        )

        if cfg.ep_bl_base > 0:
            model_path = (
                f"{cfg.saved_models_path}"
                f"model_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.ep_bl_base}_{cfg.bs_bl}.pkl"
            )
            optimizer_path = (
                f"{cfg.saved_models_path}"
                f"optimizer_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.ep_bl_base}_{cfg.bs_bl}.pkl"
            )
            scheduler_path = (
                f"{cfg.saved_models_path}"
                f"scheduler_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.ep_bl_base}_{cfg.bs_bl}.pkl"
            )

            if (
                os.path.exists(model_path)
                and os.path.exists(optimizer_path)
                and os.path.exists(scheduler_path)
                and not cfg.force
            ):
                model.load_state_dict(torch.load(model_path, map_location=device))
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
                print(f"Loaded model trained for {cfg.ep_bl_base} epochs from saved files")
            else:
                print(f"Training model for {cfg.ep_bl_base} epochs before capturing batch‐level updates...")
                model, optimizer, scheduler, test_ASR, test_ACC = train(
                    model,
                    optimizer,
                    cfg.opt,
                    scheduler,
                    criterion,
                    poisoned_train_loader,
                    test_loader,
                    poisoned_test_loader,
                    cfg.ep_bl_base,
                    cfg.global_seed,
                    device,
                    cfg.training_mode
                )
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                torch.save(scheduler.state_dict(), scheduler_path)
                print(f"Model trained for {cfg.ep_bl_base} epochs and saved")

        # Select suspected indices
        poison_amount = len(poison_indices)
        ignore_set = set(poison_indices)
        random_sus_idx = get_random_poison_idx(
            cfg.pr_sus,
            ignore_set,
            poison_indices,
            target_class_indices,
            poison_amount,
            cfg.global_seed
        )
        print(
            "Suspected samples length:", len(random_sus_idx),
            "Poison‐ratio trg:", cfg.pr_tgt,
            "Suspected %:", cfg.pr_sus  
        )
        
        csv_file = init_experiment_folder(cfg)  

        # Capture batch‐level weight updates
        important_features = capture_first_level_multi_epoch_batch_sample_weight_updates(
            random_sus_idx,
            model,
            orig_model,
            optimizer,
            cfg.opt,
            scheduler,
            criterion,
            cfg.ep_bl,
            cfg.lr,
            poisoned_train_loader,
            test_loader,
            poisoned_test_loader,
            cfg.target_class,
            cfg.sample_from_test,
            cfg.attack,
            device,
            cfg.global_seed,
            cfg.results_path  + cfg.exp,
            cfg.training_mode,
            k=cfg.k_1
        )

        with open(
            cfg.prov_path
            + f"important_features_single_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.pr_sus}_{cfg.bs_bl}_k_{cfg.k_1}.pkl",
            "wb"
        ) as f:
            pickle.dump(important_features, f)
        print("Important features shape:", important_features.shape)


        update_results(
            csv_file,
            epochs=cfg.ep_bl,
            **{
                "Batch-level |features|": important_features.shape[0],
            }
        )


    if cfg.sample_level:
        logger.info("Starting sample‐level provenance capture...")
        poisoned_train_loader, test_loader, poisoned_test_loader, target_class_indices = get_loaders_from_dataset(
            poisoned_train_dataset,
            test_dataset,
            poisoned_test_dataset,
            cfg.bs_sl,
            cfg.target_class
        )

        model = copy.deepcopy(orig_model).to(device)

        # Set up optimize
        if cfg.opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        elif cfg.opt == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=5e-4,
                dampening=0,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.opt}")

        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.ep_sl
        )

        model_path = (
            f"{cfg.saved_models_path}"
            f"model_{cfg.attack}_{cfg.dataset}_{cfg.eps}_"
            f"{cfg.pr_tgt}_{cfg.ep_sl_base}_{cfg.bs_bl}.pkl"
        )
        optimizer_path = (
            f"{cfg.saved_models_path}"
            f"optimizer_{cfg.attack}_{cfg.dataset}_{cfg.eps}_"
            f"{cfg.pr_tgt}_{cfg.ep_sl_base}_{cfg.bs_bl}.pkl"
        )
        scheduler_path = (
            f"{cfg.saved_models_path}"
            f"scheduler_{cfg.attack}_{cfg.dataset}_{cfg.eps}_"
            f"{cfg.pr_tgt}_{cfg.ep_sl_base}_{cfg.bs_bl}.pkl"
        )

        if cfg.ep_sl_base > 0:
            if (
                os.path.exists(model_path)
                and os.path.exists(optimizer_path)
                and os.path.exists(scheduler_path)
                and not cfg.force
            ):
                model.load_state_dict(torch.load(model_path, map_location=device))
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
                print(f"Loaded model trained for {cfg.ep_sl_base} epochs")
            else:
                print(f"Training for {cfg.ep_sl_base} epochs before capturing sample‐level updates...")
                model, optimizer, scheduler, train_ACC, test_ACC, clean_ACC, target_ACC = train(
                    model,
                    optimizer,
                    cfg.opt,
                    scheduler,
                    criterion,
                    poisoned_train_loader,
                    test_loader,
                    poisoned_test_loader,
                    cfg.ep_sl_base,
                    cfg.global_seed,
                    device,
                    cfg.training_mode
                )
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                torch.save(scheduler.state_dict(), scheduler_path)
                print(f"Model trained for {cfg.ep_sl_base} epochs and saved")

        # Select a random set of suspected indices
        poison_amount = len(poison_indices)
        random_sus_idx = get_random_poison_idx(
            cfg.pr_sus,
            set(poison_indices),
            poison_indices,
            target_class_indices,
            poison_amount,
            cfg.global_seed
        )
        print("Suspected samples length:", len(random_sus_idx))

        # Load previously computed important features
        feats_path = (
            f"{cfg.prov_path}"
            f"important_features_single_{cfg.attack}_{cfg.dataset}_{cfg.eps}_"
            f"{cfg.pr_tgt}_{cfg.pr_sus}_{cfg.bs_sl}_k_{cfg.k_1}.pkl"
        )
        with open(feats_path, "rb") as f:
            important_features = pickle.load(f)

        print("Important features shape:", important_features.shape)
        print(
            "Suspected samples length:", len(random_sus_idx),
            "Poison-ratio trg:", cfg.pr_tgt,
            "Suspected %:", cfg.pr_sus
        )

        # Capture sample-level weight updates
        sus_diff, clean_diff, sus_inds, clean_inds = capture_sample_level_weight_updates_idv(
            random_sus_idx,
            model,
            orig_model,
            optimizer,
            cfg.opt,
            scheduler,
            criterion,
            cfg.ep_sl,
            cfg.lr,
            poisoned_train_loader,
            test_loader,
            poisoned_test_loader,
            important_features,
            cfg.target_class,
            cfg.sample_from_test,
            cfg.attack,
            device,
            cfg.global_seed,
            cfg.results_path,
            cfg.training_mode,
            k=cfg.k_1,
        )
        print(
            "Shape of suspected updates:", sus_diff.shape,
            "Shape of clean updates:", clean_diff.shape
        )

        # Save the results
        base = (
            f"{cfg.prov_path}"
            f"sus_diff_single_{cfg.attack}_{cfg.dataset}_{cfg.eps}_"
            f"{cfg.pr_tgt}_{cfg.pr_sus}_{cfg.bs_sl}_{cfg.ep_sl}_"
            f"{cfg.ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl"
        )
        with open(base, "wb") as f:
            pickle.dump(sus_diff, f)
        base = base.replace("sus_diff_single", "clean_diff_single")
        with open(base, "wb") as f:
            pickle.dump(clean_diff, f)

        inds_path = base.replace("clean_diff_single", "sus_inds_single")
        with open(inds_path, "wb") as f:
            pickle.dump(sus_inds, f)
        inds_path = inds_path.replace("sus_inds_single", "clean_inds_single")
        with open(inds_path, "wb") as f:
            pickle.dump(clean_inds, f)

        del sus_diff, clean_diff
            
            

    if cfg.score_samples:
        logger.info("Starting scoring samples for poisoning...")
        ignore_set = set(poison_indices)
        start_time = time.time()

        suffix = (
            f"sus_diff_single_{cfg.attack}_{cfg.dataset}_{cfg.eps}_"
            f"{cfg.pr_tgt}_{cfg.pr_sus}_{cfg.bs_sl}_{cfg.ep_sl}_"
            f"{cfg.ep_sl_base}_sample_level_idv_k_1_aug_z_1.pkl"
        )
        with open(os.path.join(cfg.prov_path, suffix), "rb") as f:
            sus_diff = pickle.load(f)

        suffix = suffix.replace("sus_diff_single", "clean_diff_single")
        with open(os.path.join(cfg.prov_path, suffix), "rb") as f:
            clean_diff = pickle.load(f)

        suffix = suffix.replace("clean_diff_single", "sus_inds_single")
        with open(os.path.join(cfg.prov_path, suffix), "rb") as f:
            sus_inds = pickle.load(f)

        suffix = suffix.replace("sus_inds_single", "clean_inds_single")
        with open(os.path.join(cfg.prov_path, suffix), "rb") as f:
            clean_inds = pickle.load(f)

        # For mixed attacks, restore the full poison_indices mapping
        if cfg.attack in {"narcissus_lc", "narcissus_lc_sa"}:
            poison_indices = poison_indices_all
            

        print(
            "Shape of suspected updates:", sus_diff.shape,
            "Shape of clean updates:", clean_diff.shape,
            "Suspected indices:", np.array(sus_inds).shape,
            "Clean indices:", np.array(clean_inds).shape
        )
        random_sus_idx = np.unique(sus_inds)
        random_clean_sus_idx = list(set(random_sus_idx) - set(poison_indices))

        print(
            "Poison count:", len(poison_indices),
            "Suspected count:", len(random_sus_idx),
            "Clean‐suspected count:", len(random_clean_sus_idx)
        )
        assert set(random_clean_sus_idx) == set(random_sus_idx) - set(poison_indices)

        # Score Suspected samples
        indexes_to_exclude, tpr_kmeans, fpr_kmeans, tpr_gaussian, fpr_gaussian = score_poisoned_samples( 
            sus_diff,
            clean_diff,
            clean_inds,
            sus_inds,
            poison_indices,
            random_clean_sus_idx,
            cfg.groups,
            cfg.dataset,
            cfg.cv_model,
            cfg.ep_sl,
            cfg.global_seed,
            device,
            cfg.pr_tgt,
            cfg.pr_sus,
            cfg.attack,
            os.path.join(cfg.results_path, cfg.exp),
            cfg.custom_threshold,
            cfg.threshold_type, 
            cfg.k_2 
        )

        print(
            "Length of indexes to exclude:", len(indexes_to_exclude),
            "True poisons excluded:", len(set(indexes_to_exclude) & set(poison_indices))
        )

        # Save the selected indices to exclude
        excl_name = (
            f"indexes_to_exclude_{cfg.attack}_{cfg.dataset}_"
            f"{cfg.eps}_{cfg.pr_tgt}_{cfg.pr_sus}.pkl"
        )
        with open(os.path.join(cfg.prov_path, excl_name), "wb") as f:
            pickle.dump(indexes_to_exclude, f)

        print("Time taken for analysis:", time.time() - start_time)

        # Save the results
        csv_file = init_experiment_folder(cfg)
        update_results(
            csv_file,
            epochs=cfg.ep_sl,
            **{
                "TPR KMeans": np.round(tpr_kmeans * 100, 2),
                "FPR KMeans": np.round(fpr_kmeans * 100, 2),
                "TPR Gaussian": np.round(tpr_gaussian * 100, 2),
                "FPR Gaussian": np.round(fpr_gaussian * 100, 2),
            }
        )
        

           
    if cfg.retrain:
        logger.info("Starting retraining with excluded indices...")
        # load the indices to exclude
        excl_name = f"indexes_to_exclude_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.pr_sus}.pkl"
        with open(
            os.path.join(cfg.prov_path, excl_name)
        , "rb") as f:
            indexes_to_exculde = pickle.load(f)
            print(
                "Length of indexes to exclude:",
                len(indexes_to_exculde),
                "pos_indices:",
                len(set(indexes_to_exculde) & set(poison_indices)),
            )

        # rebuild loaders without the excluded indices
        poisoned_train_loader, test_loader, poisoned_test_loader, _ = get_loaders_from_dataset(
            poisoned_train_dataset,
            test_dataset,
            poisoned_test_dataset,
            cfg.bs,
            cfg.target_class,
            indexes_to_exculde,
        )


        model = copy.deepcopy(orig_model).to(device)
        if cfg.opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        elif cfg.opt == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=5e-4,
                dampening=0,
                nesterov=True,
            )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )

        if cfg.get_result:
            model.load_state_dict(
                torch.load(
                    cfg.saved_models_path
                    + f"retrained_model_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.pr_sus}_{cfg.epochs}.pkl"
                )
            )
            test_ASR, test_ACC = evaluate_model(model, test_loader, poisoned_test_loader, criterion, device)
            
            if cfg.attack in {"narcissus_lc"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Retrain ASR Narcissus": test_ASR[-1],
                        "Retrain ASR Label Consistent": test_ASR[-2],
                        "Retrain ACC": test_ACC,
                    }
                )
            elif cfg.attack in {"narcissus_lc_sa"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Retrain ASR Narcissus": test_ASR[-1],
                        "Retrain ASR Label Consistent": test_ASR[-2],
                        "Retrain ASR Sleeper Agent": test_ASR[-3],
                        "Retrain ACC": test_ACC,
                    }
                )
            else:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Retrain ACC": test_ACC,
                        "Retrain ASR": test_ASR[-1],
                    }
                )
            
        else:
            model, optr, slr, test_ASR, test_ACC = train(
                model,
                optimizer,
                cfg.opt,
                scheduler,
                criterion,
                poisoned_train_loader,
                test_loader,
                poisoned_test_loader,
                cfg.epochs,
                cfg.global_seed,
                device,
                cfg.training_mode,
            )
            torch.save(
                model.state_dict(),
                cfg.saved_models_path
                + f"retrained_model_{cfg.attack}_{cfg.dataset}_{cfg.eps}_{cfg.pr_tgt}_{cfg.pr_sus}_{cfg.epochs}.pkl",
            )

        
            if cfg.attack in {"narcissus_lc"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Retrain ASR Narcissus": test_ASR[-1],
                        "Retrain ASR Label Consistent": test_ASR[-2],
                        "Retrain ACC": test_ACC,
                    }
                )
            elif cfg.attack in {"narcissus_lc_sa"}:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Retrain ASR Narcissus": test_ASR[-1],
                        "Retrain ASR Label Consistent": test_ASR[-2],
                        "Retrain ASR Sleeper Agent": test_ASR[-3],
                        "Retrain ACC": test_ACC,
                    }
                )
            else:
                csv_file = init_experiment_folder(cfg)   
                update_results(
                    csv_file,
                    epochs=cfg.epochs,
                    **{
                        "Retrain ACC": test_ACC[-1],
                        "Retrain ASR": test_ASR[-1],
                    }
                )

        
    
if __name__ == '__main__':
    main()