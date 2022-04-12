
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

    training_set = MNIST(
        partition="train",
    )
    test_set = MNIST(
        partition="test",
    )

    validation_set = None
    return training_set, validation_set, test_set


class MNIST(datasets.MNIST):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        if "root" in kwargs:
            root = kwargs["root"]
        else:
            root = "./data"

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        if partition == "train":
            train = True
        elif partition == "test":
            train = False
        else:
            raise NotImplementedError(
                "The dataset partition {} does not exist".format(partition)
            )

        super(MNIST, self).__init__(
            root=root, train=train, transform=transform, download=True
        )


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
