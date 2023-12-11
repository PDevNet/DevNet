"""
Adapted from https://github.com/patrick-kidger/NeuralCDE/blob/758d3a7134e3a691013e5cc6b7f68f277e9e6b69/experiments/datasets/speech_commands.py
"""
import os
import pathlib
import urllib.request
import tarfile
import torch
import torchaudio
import ml_collections
from typing import Tuple
from .utils import normalise_data, split_data, load_data, save_data
from torch.utils.data import Dataset, DataLoader


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    training_set = SpeechCommands(
        partition="train",
        mfcc=config.mfcc
    )
    validation_set = SpeechCommands(
        partition="val",
        mfcc=config.mfcc
    )
    test_set = SpeechCommands(
        partition="test",
        mfcc=config.mfcc
    )

    training_loader = DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dataloaders = {"train": training_loader, "validation": val_loader}
    return dataloaders, test_loader

    return dataloaders, test_loader


class SpeechCommands(torch.utils.data.TensorDataset):
    def __init__(
        self,
        partition: int,
        **kwargs,
    ):
        mfcc = kwargs["mfcc"]

        self.root = pathlib.Path("./SpeechCommands/data")
        base_loc = self.root / "processed_data"

        if mfcc:
            data_loc = base_loc / "mfcc"
        else:
            data_loc = base_loc / "raw"

        if os.path.exists(data_loc):
            pass
        else:
            self.download()
            train_X, val_X, test_X, train_y, val_y, test_y = self._process_data(
                mfcc)

            if not os.path.exists(base_loc):
                os.mkdir(base_loc)
            if not os.path.exists(data_loc):
                os.mkdir(data_loc)
            save_data(
                data_loc,
                train_X=train_X,
                val_X=val_X,
                test_X=test_X,
                train_y=train_y,
                val_y=val_y,
                test_y=test_y,
            )

        X, y = self.load_data(data_loc, partition)

        super(SpeechCommands, self).__init__(X, y)

    def download(self):
        root = self.root
        base_loc = root / "SpeechCommands"
        loc = base_loc / "speech_commands.tar.gz"
        if os.path.exists(loc):
            return
        if not os.path.exists(root):
            os.mkdir(root)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve(
            "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", loc
        )  # TODO: Add progress bar
        with tarfile.open(loc, "r") as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, base_loc)

    def _process_data(self, mfcc):
        base_loc = self.root / "SpeechCommands"
        X = torch.empty(34975, 16000, 1)
        y = torch.empty(34975, dtype=torch.long)

        batch_index = 0
        y_index = 0
        for foldername in (
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ):
            loc = base_loc / foldername
            for filename in os.listdir(loc):
                audio, _ = torchaudio.load(
                    loc / filename, channels_first=False, normalize=False
                )  # for forward compatbility if they fix it
                audio = (
                    audio / 2 ** 15
                )  # Normalization argument doesn't seem to work so we do it manually.

                # A few samples are shorter than the full length; for simplicity we discard them.
                if len(audio) != 16000:
                    continue

                X[batch_index] = audio
                y[batch_index] = y_index
                batch_index += 1
            y_index += 1
        assert batch_index == 34975, "batch_index is {}".format(batch_index)

        # If MFCC, then we compute these coefficients.
        if mfcc:
            X = torchaudio.transforms.MFCC(
                log_mels=True, n_mfcc=20, melkwargs=dict(n_fft=200, n_mels=64)
            )(X.squeeze(-1)).detach()
            # X is of shape (batch=34975, channels=20, length=161)
        else:
            X = X.unsqueeze(1).squeeze(-1)
            # X is of shape (batch=34975, channels=1, length=16000)

        # Normalize data
        X = normalise_data(X, y)

        train_X, val_X, test_X = split_data(X, y)
        train_y, val_y, test_y = split_data(y, y)

        return (
            train_X,
            val_X,
            test_X,
            train_y,
            val_y,
            test_y,
        )

    @staticmethod
    def load_data(data_loc, partition):

        tensors = load_data(data_loc)
        if partition == "train":
            X = tensors["train_X"]
            y = tensors["train_y"]
        elif partition == "val":
            X = tensors["val_X"]
            y = tensors["val_y"]
        elif partition == "test":
            X = tensors["test_X"]
            y = tensors["test_y"]
        else:
            raise NotImplementedError(
                "the set {} is not implemented.".format(set))

        return X, y
