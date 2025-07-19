# **PoisonSpot: Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking**
<img width="417" height="337" alt="PoisonSpot-Logo" src="https://github.com/user-attachments/assets/9f69cf04-2390-4575-9220-97e9a2a97688" />


This repository contains the implementation of the paper **[PoisonSpot: Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking](https://github.com/um-dsp/PoisonSpot/blob/main/PoisonSpot-CCS2025.pdf)**. PoisonSpot is a novel system that precisely detects clean-label backdoor attacks by using fine-grained training provenance tracking. Inspired by dynamic taint tracking, PoisonSpot uses fine-grained training provenance tracker that: (1) tags & traces the impact of every single training sample on model updates, (2) probabilistically scores suspect samples based on their lineage of impact on model weights, and (3) separates the clean from the poisonous before retraining a model. 

<img width="1007" height="363" alt="PoisonSpot-Overview" src="https://github.com/user-attachments/assets/38607bac-03b2-4b8f-aa68-dc146be3f996" />


---


## Steps of PoisonSpot
1. **Poisoned Model Training (`poisoned_training`)**
   - Train a model on a dataset with a specified percentage of poisoned samples. 
2. **Batch Level Provenance Capture (`batch_level`)**
   - Capture the batch-level provenance data using the trained model to get important features.
3. **Sample Level Provenance Capture (`sample_level`)**
   - Capture the sample-level provenance data using the trained model and the important features. 
4. **Poisoning Score Attribution (`score_samples`)**
   - Score the suspected samples using the captured sample-level provenance data.
5. **Retraining (`retrain`)**
   - Retrain and evaluate the model by removing the predicted poisoned samples from the training set.


## Overview of the folder structure
```
PoisonSpot/
├── configs/                   # configurations for different attacks, datasets, and other hyper-parameters. 
├── src/                       # Source code
│   ├── attacks/               # Attack implementations
│   │   ├── HiddenTriggerBackdoor/
│   │   ├── Labelconsistent/
│   │   ├── mixed/
│   │   ├── Narcissus/
│   │   └── Sleeperagent/
│   ├── data/                  # Stores downloaded poisoned-datasets.
│   ├── helpers/               
│   │   ├── data.py            # Data loading function
│   │   ├── provenance.py      # Functions for capturing batch-level and sample-level provenance data 
│   │   ├── scoring.py         # Score samples for poisoning 
│   │   └── train.py           # model training funciton 
│   ├── models/                # Model definitions (e.g., ResNet)
│   ├── results/               # Stores results for each experiment, generated figures and config logs 
│   ├── saved_models/          # Model checkpoints for clean training, poisoned training, and retraining. 
│   ├── temp_folder/           # Temporary folder to save model during provenance capture
│   └── Training_Prov_Data/    # Stores captured provenance data (batch-level & sample-level) 
├── main.py                    # Main entry point code for all the steps 
├── README.md                  # Instructions for running the PoisonSpot experiments
└── requirements.txt           # Python dependencies
```


## Arguments

| Argument             | Description                                                    | Default Value                           |
|----------------------|----------------------------------------------------------------|-----------------------------------------|
| `batch_level`        | Capture batch-level provenance data                            | `True`                                  |
| `clean_training`     | Perform clean training by removing the suspected samples       | `False`                                 |
| `poisoned_training`  | Perform training using the suspected samples                   | `False`                                  |
| `sample_level`       | Capture sample-level provenance data                           | `True`                                  |
| `score_samples`      | Score suspected samples based on the sample-level provenance data | `True`                               |
| `retrain`            | Retrain the model by excluding predicted poisoned samples      | `True`                                  |
| `pr_sus`             | Percentage of poisoned data in the suspected set (%)           | `50`                                  |
| `ep_bl`              | Training epochs for batch-level provenance capture             | `1`                                   |
| `ep_bl_base`         | Epoch number to start batch-level provenance capture           | `200`                                  |
| `ep_sl`              | Training epochs for sample-level provenance capture            | `10`                                   |
| `ep_sl_base`         | Epoch number to start sample-level provenance capture          | `200`                                  |
| `pr_tgt`             | Percentage of poisoned data in the target set (%)              | `10`                                   |
| `bs_sl`              | Batch size for sample-level provenance capture                 | `128`                                  |
| `bs_bl`              | Batch size for batch-level provenance capture                  | `128`                                  |
| `bs`                 | Batch size for clean training, poisoned training, and retraining | `128`                               |
| `eps`                | Perturbation budget (`eps/255`)                                | `16`                                   |
| `vis`                | Pixel value for Label consistent attack                                                  | `255`                                 |
| `target_class`       | Target class for the attack                                    | `2`                                    |
| `source_class`       | Source class for the attack (mainly relevant for `sa`)                                    | `0`                                    |
| `dataset`            | Dataset to use for the experiment. (`CIFAR10`, `slt10`,`imagenet`, )                              | `CIFAR10`                            |
| `attack`             | Attack to use for the experiment, (`lc`,`narcissus`,`sa`,`ht`)                              | `lc`                                 |
| `model`              | Model to use for the experiment, (`ResNet18`,`CustomCNN`,`BasicResNet`,`CustomResNet18`,`ViT`)                                | `ResNet18`                           |
| `dataset_dir`        | Root directory for the datasets                                | `./data/`                            |
| `clean_model_path`   | Path to the trained clean model used for fine-tuning           | `./saved_models/resnet18_200_clean.pth` |
| `saved_models_path`  | Path to save the trained models                                | `./saved_models/`                    |
| `global_seed`        | Global seed for the experiment                                 | `545`                                  |
| `gpu_id`             | GPU device ID to use for the experiment                        | `0`                                    |
| `lr`                 | Learning rate for the experiment                               | `0.1`                                  |
| `results_path`       | Path to save the figures                                       | `./results/`                         |
| `prov_path`          | Path to save the provenance data                               | `./Training_Prov_Data/`              |
| `epochs`             | Number of epochs for clean training, poisoned training, and retraining | `200`                              |
| `scenario`           | Scenario to use for the experiment (`fine_tune`,`from_scratch`) | `from_scratch`                   |
| `get_result`         | Get results from previous runs                                 | `False`                                |
| `force`              | Force the run overwriting previous model checkpoints           | `False`                                |
| `custom_threshold`   | Custom threshold for scoring suspected samples                 | `0.5`                                  |
| `Threshold_type`     | Kmeans, Gaussian, or custom threshold choice                   | `Kmeans`                               |
| `k_1`                | First phase threshold                                          | `1`                                    |
| `k_2`                | Second phase threshold                                         | `0.0001`                               |
| `sample_from_test`   | Sample from the test set                                       | `False`                                |
| `cv_model`           | Model to use for cross-validation (`RandomForest`, `LogisticRegression`, `LinearSVM`, `KernelSVM` , `MLP`)                              | `RandomForest`                       |
| `groups`             | Number of groups to use for cross-validation                   | `5`                                    |
| `opt`                | Optimizer to use for the experiment (`sgd`, `adam`)            | `sgd`                                |
| `random`             | Random trigger for Sleeper-Agent attack (`True`,`False`)    | `False`                                |
| `training_mode`      | Training mode for the experiment (`true`, `false`)           | `true`                                 |


---

## Hardware and Software Setup

### Hardware (tested on)

- **CPU:** AMD Ryzen Threadripper 7960X (24 cores / 48 threads, 545 – 5362 MHz)  
- **RAM:** 251 GiB total 
- **GPU:** NVIDIA GeForce RTX 4090 (24 564 MiB)  

### Software (tested on)

- **Python:** 3.10.12  
- **CUDA version:** 12.4  




## Installation

1. **Obtain the code:**
   - **From GitHub:**  
     ```bash
     git clone https://github.com/Philenku/PoisonSpot.git
     ```
   - **From Zenodo:**  
     ```bash
     unzip PoisonSpot-v1.0.3.zip   # Download the zip file from zenodo and adjust filename as needed. 
     ```

2. **(Optional) Create & activate a virtual environment:**
   ```bash
   python -m venv venv
   ```
   ```bash
   source venv/bin/activate  # linux 
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Enter the PoisonSpot directory:**
   ```bash
   cd PoisonSpot # or go to the desired folder name
   ```



# Usage Instructions

1. **Prepare your config file**  
   Pick an example configuration file (YAML) from `configs/`, then edit the arguments such as`pr_tgt` and `pr_sus` to suit your experiment:
   For example, to run a full pipeline Label-Consistent attack on CIFAR-10 from scratch, you can use the following configuration file:

   ```bash
   configs/config_lc_cifar_10.yaml
   ```

2. **Run the full pipeline**
   You can run the full pipeline using the `main.py` script. The script will execute all steps of PoisonSpot based on the provided configuration file.

   ```bash
   python3 main.py -c configs/config_lc_cifar_10.yaml
   ```
3. **Alternative Usage method** 
   To run several configuration files in one go, use the shell script in the scripts/ directory. List the configs you want inside the script, make it executable, and then run it:

   ```bash
   chmod +x scripts/run.sh
   ```
   ```bash 
   ./scripts/run.sh
   ```

---

## Configuration Files
This repository includes several custom configuration files for different attacks and datasets. Each configuration file is designed to run a specific attack on a dataset with predefined parameters. These configuration files are located in the `configs/` directory. You can modify these files to adjust parameters such as the percentage of poisoned samples, batch size, learning rate, and more. The configuration files are written in YAML format. You can easily edit them to customize your experiments. 

Below are the custom configuration files included in this repository, with a brief description of each.

### Hidden-Trigger (ht)
| Filename                               | Dataset   | Scenario      | pr_tgt (%) | pr_sus (%) | Result location in the paper |
|----------------------------------------|-----------|---------------|-----------:|-----------:|-----------------------------|
| `configs/config_ht_cifar_10.yaml`           | CIFAR-10  | fine_tune  |         50 |         50 | Table 6  |
| `configs/config_ht_slt_10.yaml`             | STL-10    | fine_tune  |         50 |         50 | Table 17 |
| `configs/config_ht_imagenet_fine_tune.yaml` | ImageNet  | fine_tune  |         50 |         50 | Table 17 |


###  Sleeper-Agent on CIFAR-10

| Filename                                   | Dataset   | Scenario       | pr_tgt (%) | pr_sus (%) | Result location in the paper   |
|--------------------------------------------|-----------|----------------|-----------:|-----------:|--------------------------------|
| `configs/config_sa_cifar_10.yaml`          | CIFAR-10  | from_scratch   |         10 |         10 |      Table 8                   |
| `configs/config_sa_cifar_10_10_25.yaml`    | CIFAR-10  | from_scratch   |         10 |         25 |      Table 8                     |
| `configs/config_sa_cifar_10_10_50.yaml`    | CIFAR-10  | from_scratch   |         10 |         50 |      Table 7/8                           |
| `configs/config_sa_cifar_10_10_75.yaml`    | CIFAR-10  | from_scratch   |         10 |         75 |      Table 8                             |
| `configs/config_sa_cifar_10_10_100.yaml`   | CIFAR-10  | from_scratch   |         10 |        100 |      Table 8                           |
| `configs/config_sa_cifar_10_20_50.yaml`    | CIFAR-10  | from_scratch   |         20 |         50 |      Table 7                          |
| `configs/config_sa_cifar_10_30_50.yaml`    | CIFAR-10  | from_scratch   |         30 |         50 |      Table 7                            |
| `configs/config_sa_cifar_10_fine_tune.yaml`| CIFAR-10  | fine_tune      |         50 |         50 |      Table 6                          |
| `configs/config_sa_slt_10.yaml`            | CIFAR-10  | fine_tune      |          50 |          50 |      Table 17                       |

---

### Label-Consistent on CIFAR-10

| Filename                                   | Dataset   | Scenario       | pr_tgt (%) | pr_sus (%) | Result location in the paper  |
|--------------------------------------------|-----------|----------------|-----------:|-----------:|------------------------|
| `configs/config_lc_cifar_10.yaml`          | CIFAR-10  | from_scratch   |         10 |         10 | Table 8    |
| `configs/config_lc_cifar_10_10_25.yaml`    | CIFAR-10  | from_scratch   |         10 |         25 | Table 8                        |
| `configs/config_lc_cifar_10_10_50.yaml`    | CIFAR-10  | from_scratch   |         10 |         50 | Table 7/8                        |
| `configs/config_lc_cifar_10_10_75.yaml`    | CIFAR-10  | from_scratch   |         10 |         75 | Table 8                        |
| `configs/config_lc_cifar_10_10_100.yaml`   | CIFAR-10  | from_scratch   |         10 |        100 | Table 8                        |
| `configs/config_lc_cifar_10_20_50.yaml`    | CIFAR-10  | from_scratch   |         20 |         50 | Table 7                       |
| `configs/config_lc_cifar_10_30_50.yaml`    | CIFAR-10  | from_scratch   |         30 |         50 | Table 7                       |
| `configs/config_lc_cifar_10_40_50.yaml`    | CIFAR-10  | from_scratch   |         40 |         50 | Table 12                      |
| `configs/config_lc_cifar_10_50_50.yaml`    | CIFAR-10  | from_scratch   |         50 |         50 | Table 12                       |
| `configs/config_lc_cifar_10_75_75.yaml`    | CIFAR-10  | from_scratch   |         75 |         75 | Table 13                       |
| `configs/config_lc_cifar_10_100_100.yaml`  | CIFAR-10  | from_scratch   |        100 |         100| Table 13                       |
| `configs/config_lc_cifar_10_eps_2.yaml`    | CIFAR-10  | from_scratch   |         10 |         50 | Table 9  |
| `configs/config_lc_cifar_10_eps_4.yaml`    | CIFAR-10  | from_scratch   |         10 |         50 | Table 9  |

---

### Narcissus on CIFAR-10

| Filename                                   | Dataset   | Scenario       | pr_tgt (%) | pr_sus (%) | Result location in the paper                   |
|--------------------------------------------|-----------|----------------|-----------:|-----------:|------------------------|
| `configs/config_narcissus_cifar_10.yaml`       | CIFAR-10  | from_scratch   |         10 |         10 | Table 8                        |
| `configs/config_narcissus_cifar_10_10_25.yaml` | CIFAR-10  | from_scratch   |         10 |         25 | Table 8                        |
| `configs/config_narcissus_cifar_10_10_50.yaml` | CIFAR-10  | from_scratch   |         10 |         50 | Table 7/8                      |
| `configs/config_narcissus_cifar_10_10_75.yaml` | CIFAR-10  | from_scratch   |         10 |         75 | Table 8                        |
| `configs/config_narcissus_cifar_10_10_100.yaml`| CIFAR-10  | from_scratch   |         10 |        100 | Table 8                        |
| `configs/config_narcissus_cifar_10_20_50.yaml` | CIFAR-10  | from_scratch   |         20 |         50 | Table 7                        |
| `configs/config_narcissus_cifar_10_30_50.yaml` | CIFAR-10  | from_scratch   |         30 |         50 | Table 7                        |
| `configs/config_narcissus_cifar_10_40_50.yaml` | CIFAR-10  | from_scratch   |         40 |         50 | Table 12                       |
| `configs/config_narcissus_cifar_10_50_50.yaml` | CIFAR-10  | from_scratch   |         50 |         50 | Table 12                       |
| `configs/config_narcissus_cifar_10_75_75.yaml` | CIFAR-10  | from_scratch   |         75 |         75 | Table 13                       |
| `configs/config_narcissus_cifar_10_100_100.yaml` | CIFAR-10 | from_scratch   |       100 |        100| Table 13                        |
| `configs/config_narcissus_cifar_10_10_50_eps_6.yaml`| CIFAR-10  | from_scratch   |         10 |         50 | Table 9                   |
| `configs/config_narcissus_cifar_10_10_50_eps_8.yaml`| CIFAR-10  | from_scratch   |         10 |         50 | Table 9                   |
| `configs/config_narcissus_cifar_10_0.5_5.yaml`  | CIFAR-10  | from_scratch   |          0.5 |       50 | Table 8                         |

### Mixed Attacks on CIFAR-10

| Filename                                            | Dataset   | Scenario      | pr_tgt (%)                           | pr_sus (%)                           | Result location in the paper  |
|-----------------------------------------------------|-----------|---------------|--------------------------------------|--------------------------------------|-----------------|
| `configs/config_mixed_lc_narcissus_cifar_10.yaml`     | CIFAR-10  | from_scratch  | LC = 10; Narcissus = 0.5             | LC = 50; Narcissus = 50              | Table 10              |
| `configs/config_mixed_lc_sa_narcissus_cifar_10.yaml`  | CIFAR-10  | from_scratch  | LC = 10; SA = 10; Narcissus = 0.5    | LC = 50; SA = 50; Narcissus = 50     | Table 11              |




## Note
- The `pr_tgt` argument specifies the percentage of poisoned samples in the target set. So, to be consistent with our paper, in which we use the percentage of poisoned samples in the whole dataset, you can divide pr_tgt by the number of classes (10). For example, if you set `pr_tgt=10`, it means you are running an experiment with a 1% poisoned sample percentage in the training set.

- The `data` folder stores downloaded poisoned datasets for the label-consistent, sleeper-agent, and hiddentrigger backdoors which saves time by avoiding the need to regenerate the poisoned datasets every time you run an experiment.
- The `saved_models` directory also contains saved poisoned models for the configs listed, so you can set `poisoned_training = False`. If you want to discard the saved checkpoint, you need to revert it to `True` to train from scratch. 
- If you encounter a high number of important features during `batch_level` training, the provenance data may be too large for your CPU, so you can increase the epochs for the batch-level capture (`ep_bl`), raise the first-phase threshold `k`, or set `poisoned_training = True` to train the model correctly before capturing the provenance.




### **Results**
- The output results are saved in `results` folder 
   ```
   src/results/experiment_<attack>_<dataset>_<pr_tgt>_<pr_sus>/results.csv
   ```
You can also specify a custom experiment name via the `exp` field in your config.

The CSV contains the following columns:

| Column                      | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| `epochs`                    | Number of epochs  for the specific step                          |
| `Clean training ACC`        | Accuracy on clean test set after clean training                  |
| `Poisoned training ACC`     | Accuracy on poisoned test set after poisoned training            |
| `Poisoned training ASR`     | Attack success rate on poisoned test set                         |
| `Batch-level \|features\|`  | Number of important features from batch level provenance capture |
| `TPR KMeans`                | True positive rate using K-means threshold   after scoring samples |
| `FPR KMeans`                | False positive rate using K-means threshold                      |
| `TPR Gaussian`              | True positive rate using Gaussian threshold              |
| `FPR Gaussian`              | False positive rate using Gaussian hreshold             |
| `Retrain ACC`               | Accuracy on clean test set after retraining    |
| `Retrain ASR`               | Attack success rate after retraining           |

- The results in the paper are mostly based on the `KMeans` threshold, but you can also use the `Gaussian` threshold for comparison if it brings a better result.
- The provenance data collected during training is saved in the `Training_Prov_Data` folder.
- Visualizations of the poison score distribution from the experiments are saved in the experiment folder under `results/experiment_<attack>_<dataset>_<pr_tgt>_<pr_sus>/` as images.



## Citation

```bibtex
@inproceedings{hailemariam2025poisonspot,
  author    = {Philemon Hailemariam, Birhanu Eshete},
  title     = {PoisonSpot: Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking},
  booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
  year      = {2025}
}
```
---

