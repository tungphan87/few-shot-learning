import urllib.request
from urllib.error import HTTPError

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision import transforms as T
from torchvision.datasets import CIFAR100, SVHN
from src.data.constants import (
    DATASET_PATH,
    N_WAY,
    K_SHOT,
)
from src.data.make_dataset import ImageDataSet
from src.data.meta_sampler import BatchSampler


# set seed
torch.manual_seed(0)


def load_data_CIFAR(DATASET_PATH):
    CIFAR_train_set = CIFAR100(
        root=DATASET_PATH, train=True, download=True, transform=transforms.ToTensor()
    )
    CIFAR_test_set = CIFAR100(
        root=DATASET_PATH, train=False, download=True, transform=transforms.ToTensor()
    )

    CIFAR_all_images = np.concatenate([CIFAR_train_set.data, CIFAR_test_set.data])
    CIFAR_all_targets = torch.LongTensor(
        CIFAR_train_set.targets + CIFAR_test_set.targets
    )

    return CIFAR_all_images, CIFAR_all_targets


def split_train_val_test_classes():
    classes = torch.randperm(100)
    train_classes, val_classes, test_classes = (
        classes[:80],
        classes[80:90],
        classes[90:],
    )

    classes = torch.randperm(100)
    train_classes, val_classes, test_classes = (
        classes[:80],
        classes[80:90],
        classes[90:],
    )

    return train_classes, val_classes, test_classes


def dataset_from_labels(imgs, targets, class_set, **kwargs):
    # good trick here using broadcasting
    class_mask = (targets[:, None] == class_set[None, :]).any(dim=1)
    return imgs[class_mask], targets[class_mask]


def create_transforms(split, data_means, data_stds):
    if split == "train" or split == "val":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize(data_means, data_stds),
            ]
        )
    else:
        return T.Compose([T.ToTensor(), T.Normalize(data_means, data_stds)])


def load_data(dataset, N_way, K_shot, include_query, shuffle):
    dataset_loader = data.DataLoader(
        dataset,
        batch_sampler=BatchSampler(
            dataset.targets,
            include_query=include_query,
            N_way=N_way,
            K_shot=K_shot,
            shuffle=shuffle,
        ),
        num_workers=4,
    )

    return dataset_loader


def split_train_val_test_sets():
    
    CIFAR_all_images, CIFAR_all_targets = load_data_CIFAR(DATASET_PATH)
    train_classes, val_classes, test_classes = split_train_val_test_classes()
    
    train_imgs, train_targets = dataset_from_labels(
        CIFAR_all_images, CIFAR_all_targets, train_classes
    )
    val_imgs, val_targets = dataset_from_labels(
        CIFAR_all_images, CIFAR_all_targets, val_classes
    )
    test_imgs, test_targets = dataset_from_labels(
        CIFAR_all_images, CIFAR_all_targets, test_classes
    )

    data_means = (train_imgs / 255.0).mean(axis=(0, 1, 2))
    data_stds = (train_imgs / 255.0).std(axis=(0, 1, 2))

    train_set = ImageDataSet(
        train_imgs,
        train_targets,
        transforms=create_transforms("train", data_means, data_stds),
    )

    train_set = ImageDataSet(
        train_imgs,
        train_targets,
        transforms=create_transforms("train", data_means, data_stds),
    )
    val_set = ImageDataSet(
        val_imgs,
        val_targets,
        transforms=create_transforms("val", data_means, data_stds),
    )
    test_set = ImageDataSet(
        test_imgs,
        test_targets,
        transforms=create_transforms("test", data_means, data_stds),
    )

    train_data_loader = load_data(
        train_set, N_WAY, K_SHOT, include_query=True, shuffle=True
    )
    val_data_loader = load_data(
        val_set, N_WAY, K_SHOT, include_query=True, shuffle=True
    )
    test_data_loader = load_data(
        test_set, N_WAY, K_SHOT, include_query=False, shuffle=True
    )

    return train_data_loader, val_data_loader, test_data_loader
