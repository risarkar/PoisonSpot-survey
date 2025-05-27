# src/__init__.py
# ---------------

# attacks
from .attacks.Labelconsistent.generate_poison_lc import (
    get_lc_cifar10_poisoned_data,
    get_lc_image_net_poisoned_data,
)
from .attacks.Narcissus.generate_poison_narcissus import (
    get_narcissus_cifar10_poisoned_data,
)
from .attacks.Sleeperagent.generate_poison_sa import (
    get_sa_cifar10_poisoned_data,
    get_sa_slt_10_poisoned_data,
)
from .attacks.HiddenTriggerBackdoor.generate_poison_hidden_trigger import (
    get_ht_cifar10_poisoned_data,
    get_ht_stl10_poisoned_data,
    get_ht_imagenet_poisoned_data,
)
from .attacks.mixed.mixed_attacks import (
    get_lc_narcissus_cifar_10_poisoned_data,
    get_lc_narcissus_sa_cifar_10_poisoned_data,
)

# models
from .models.resnet import ResNet
from .models.custom_resnet18 import CustomResNet18
from .models.dnn import DNN
from .models.custom_cnn import CustomCNN
from .models.custom_vit import CustomViT

# utils
from .utils.util import AverageMeter

from .helpers.data import get_loaders_from_dataset, get_random_poison_idx
from .helpers.train import train, evaluate_model
from .helpers.provenance import (
    capture_first_level_multi_epoch_batch_sample_weight_updates,
    capture_sample_level_weight_updates_idv,
)
from .helpers.scoring import train_prov_data_custom, score_poisoned_samples, get_diff

__all__ = [
    # attacks
    "get_lc_cifar10_poisoned_data",
    "get_lc_image_net_poisoned_data",
    "get_narcissus_cifar10_poisoned_data",
    "get_sa_cifar10_poisoned_data",
    "get_sa_slt_10_poisoned_data",
    "get_ht_cifar10_poisoned_data",
    "get_ht_stl10_poisoned_data",
    "get_ht_imagenet_poisoned_data",
    "get_lc_narcissus_cifar_10_poisoned_data",
    "get_lc_narcissus_sa_cifar_10_poisoned_data",

    # models
    "ResNet",
    "CustomResNet18",
    "CustomCNN",
    "CustomViT",
    "DNN",

    # utils
    "AverageMeter",

    # data helpers
    "get_loaders_from_dataset",
    "get_random_poison_idx",

    # training helpers
    "train",
    "evaluate_model",

    # provenance capture
    "capture_first_level_multi_epoch_batch_sample_weight_updates",
    "capture_sample_level_weight_updates_idv",

    # scoring / defence
    "train_prov_data_custom",
    "score_poisoned_samples",
    "get_diff",
]