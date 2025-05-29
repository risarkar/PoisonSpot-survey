# **PoisonSpot: Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking**


This repository contains the implementation of the paper **PoisonSpot: Precise Spotting of Clean-Label Backdoors via Fine-Grained Training Provenance Tracking**. PoisonSpot is a novel system that precisely detects clean-label backdoor attacks by using fine-grained training provenance tracking, inspired by dynamic taint tracking. PoisonSpot captures and analyzes the impact of individual training samples on model parameter updates throughout the training process. By attributing poisoning scores to suspect samples based on their impact lineage, PoisonSpot allows for accurate identification and rejection of samples carrying backdoor triggers.


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


## Arguments
Below is a list of arguments you can use with PoisonSpot and their functions:

| Argument               | Description                                                    | Default Value                      |
|------------------------|---------------------------------------------------------------|------------------------------------|
| `batch_level`        | Capture batch-level provenance data                                | `True`                                   |
| `clean_training`     | Perform clean training by removing the suspected samples      | `False`                                   |
| `poisoned_training`  | Perform training using the suspected samples                  | `True`                                   |
| `sample_level`       | Capture sample-level provenance data                              | `True`                                   |
| `score_samples`      | Score suspected samples based on the sample-level provenance data                                       | `True`                                   |
| `retrain`            | Retrain the model by excluding predicted poisoned samples              | `True`                                   |
| `pr_sus`             | Percentage of poisoned data in the suspected set (%)          | `100`                              |
| `ep_bl`              | Training epochs for batch-level provenance capture               | `10`                               |
| `ep_bl_base`         | Epoch number to start batch-level provenance capture             | `200`                              |
| `ep_sl`              | Training epochs for sample-level provenance capture                     | `10`                               |
| `ep_sl_base`         | Epoch number to start sample-level provenance capture                   | `200`                              |
| `pr_tgt`             | Percentage of poisoned data in the target set (%)                 | `10`                              |
| `bs_sl`              | Batch size for sample-level provenance capture                         | `128`                              |
| `bs_bl`              | Batch size for batch-level provenance capture                          | `128`                              |
| `bs`                 | Batch size for clean training, poisoned training, and retraining | `128`                              |
| `eps`                | perturbation budget (`eps/255`)                              | `16`                                |
| `vis`                | pixel value for label-consistent attack (0-255)              | `255`                                 |
| `target_class`       | Target class for the attack                                  | `2`                                |
| `source_class`       | Source class for the attack                                  | `0`                                |
| `dataset`            | Dataset to use for the experiment                            | `"CIFAR10"`                       |
| `attack`             | Attack to use for the experiment                             | `"lc"`                             |
| `model`              | Model to use for the experiment                              | `"ResNet18"`                      |
| `dataset_dir`        | Root directory for the datasets                              | `"./data/"`                    |
| `clean_model_path`   | Path to the trained clean model using for fine-tuning         | `'./saved_models/resnet18_200_clean.pth'` |
| `saved_models_path`  | Path to save the trained models                                      | `'./saved_models/'`                |
| `global_seed`        | Global seed for the experiment                               | `545`                              |
| `gpu_id`             | GPU device ID to use for the experiment                      | `0`                                |
| `lr`                 | Learning rate for the experiment                             | `0.1`                              |
| `results_path`       | Path to save the figures                                     | `"./results/"`                     |
| `prov_path`          | Path to save the provenance data                             | `"./Training_Prov_Data/"`          |
| `epochs`             | Number of epochs for clean training, poisoned training, and retraining       | `200`                              |
| `scenario`           | Scenario to use for the experiment (`fine_tune` or `from_scratch`) | `"from_scratch"`                   |
| `get_result`         | Get results from previous runs                               | `False`                                   |
| `force`              | Force the run overwriting previous results                   | `False`                                   |
| `threshold`          | Custom threshold for scoring suspected samples                      | `0.5`                              |
| `sample_from_test`   | Sample from the test set                                     | `False`                                   |
| `cv_model`           | Model to use for cross-validation                            | `"RandomForest"`                   |
| `groups`             | Number of groups to use for cross-validation                 | `5`                                |
| `opt`                | Optimizer to use for the experiment  (`sgd`, `adam`)              | `"sgd"`                            |
| `random`             | Random trigger for Sleeper-Agent attack (`True` or `False`) | `False`                                   |
| `training_mode`      | Training mode for the experiment (`true` or `false`)          |  `true`                                   |

---



## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_repo/PoisonSpot.git
   ```
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Move to the PoisonSpot directory:**
   ```bash
   cd PoisonSpot
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

---

## Configuration Files
This repository includes several custom configuration files for different attacks and datasets. Each configuration file is designed to run a specific attack on a dataset with predefined parameters. These configuration files are located in the `configs/` directory. You can modify these files to adjust parameters such as the percentage of poisoned samples, batch size, learning rate, and more. The configuration files are written in YAML format. You can easily edit them to customize your experiments. 

Below are the custom configuration files included in this repository, with a brief description of each:

| Filename                                          | Description                                                                                  |
|---------------------------------------------------|----------------------------------------------------------------------------------------------|
| `configs/config_lc_cifar_10.yaml`                 | Label-Consistent attack on CIFAR-10 from scratch (default: `pr_tgt=10%`, `pr_sus=50%`)       |
| `configs/config_sa_cifar_10.yaml`                 | Sleeper-Agent attack on CIFAR-10 from scratch (default: `pr_tgt=10%` `pr_sus=50%`)           |
| `configs/config_narcissus_cifar_10.yaml`          | Narcissus attack on CIFAR-10 from scratch                                                    |
| `configs/config_mixed_lc_narcissus_cifar_10.yaml` | Mixed Label-Consistent + Narcissus on CIFAR-10                                               |
| `configs/config_mixed_lc_sa_narcissus_cifar_10.yaml` | Mixed Label-Consistent + Sleeper-Agent + Narcissus on CIFAR-10                            |
| `configs/config_ht_cifar_10.yaml`                 | Hidden-Trigger attack on CIFAR-10                                                            |
| `configs/config_ht_slt_10.yaml`                   | Hidden-Trigger attack on STL-10                                                              |
| `configs/config_sa_cifar_10_fine_tune.yaml`       | Sleeper-Agent on CIFAR-10 (Fine tuning)                                                  |
| `configs/config_ht_imagenet_fine_tune.yaml`       | Hidden-Trigger attack on ImageNet                                                            |


## Note
- The pr_tgt argument specifies the percentage of poisoned samples in the target set. So, to be consistent with our paper, in which we use the percentage of poisoned samples in the whole dataset, you can divide pr_tgt by the number of classes (10). For example, if you set pr_tgt=10, it means you are running an experiment with a 1% poisoned sample rate in the training set.


### **Results**
- The output results are saved in `results` folder 
   ```
   src/results/experiment_<attack>_<dataset>_<pr_tgt>_<pr_sus>/results.csv
   ```
You can also specify a custom experiment name via the `exp` field in your config.

The CSV contains the following columns:

| Column                      | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| `epochs`                    | Number of epochs run                                             |
| `Clean training ACC`        | Accuracy on clean test set after clean training                  |
| `Poisoned training ACC`     | Accuracy on poisoned test set after poisoned training            |
| `Poisoned training ASR`     | Attack success rate on poisoned test set                         |
| `Batch-level \|features\|`  | Number of important features from batch level provenance capture |
| `TPR KMeans`                | True positive rate using K-means threshold                       |
| `FPR KMeans`                | False positive rate using K-means threshold                      |
| `TPR Gaussian`              | True positive rate using Gaussian threshold              |
| `FPR Gaussian`              | False positive rate using Gaussian hreshold             |
| `Retrain ACC`               | Accuracy on clean test set after retraining    |
| `Retrain ASR`               | Attack success rate after retraining           |

- The results in the paper are mostly based on the `KMeans` threshold, but you can also use the `Gaussian` threshold for comparison.
- The provenance data collected during training is saved in the `Training_Prov_Data` folder.
- Visualizations of the poison score distribution from the experiments are saved in the experiment folder under `results/experiment_<attack>_<dataset>_<pr_tgt>_<pr_sus>/` as images.



---

