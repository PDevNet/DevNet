
from torchvision import datasets, transforms
import torch
import ml_collections
from typing import Tuple


def dataset_constructor(
    config: ml_collections.ConfigDict,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple (training_set, validation_set, test_set)
    """

    training_set = CIFAR10(
        partition="train",
    )
    test_set = CIFAR10(
        partition="test",
    )

    validation_set = None
    return training_set, validation_set, test_set


class CIFAR10(datasets.CIFAR10):  # TODO: Documentation
    def __init__(
        self,
        partition: str,
        **kwargs,
    ):

        root = "./data"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ]
        )
        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super().__init__(root=root, train=train, transform=transform, download=True)


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    training_set, validation_set, test_set = dataset_constructor(config)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if validation_set is not None:
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        val_loader = test_loader

    dataloaders = {"train": training_loader, "validation": val_loader}

    return dataloaders, test_loader
