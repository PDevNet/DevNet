import torch
import os
import sklearn.model_selection
import pickle
import pathlib


def pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[: channel.size(0)] = channel
    return out


def subsample(X, y, subsample_rate):
    if subsample_rate != 1:
        X = X[:, :, ::subsample_rate]
    return X, y


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + ".pt")


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith(".pt"):
            tensor_name = filename.split(".")[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        # compute statistics using only training data.
        mean = train_Xi_nonan.mean()
        std = train_Xi_nonan.std()
        out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (
        train_tensor,
        testval_tensor,
        train_stratify,
        testval_stratify,
    ) = sklearn.model_selection.train_test_split(
        tensor,
        stratify,
        train_size=0.7,
        random_state=0,
        shuffle=True,
        stratify=stratify,
    )

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(
        testval_tensor,
        train_size=0.5,
        random_state=1,
        shuffle=True,
        stratify=testval_stratify,
    )
    return train_tensor, val_tensor, test_tensor


def add_time(X: torch.tensor) -> torch.tensor:
    """add time dimension to the input time sereies tensor

    Args:
        X (torch.tensor): time series tensor with shpe (N,C,T)

    Returns:
         time added time series tensor with shpe (N,C+1,T)
    """
    N, C, T = X.shape
    time = torch.arange(T).view(1, 1, T).repeat(N, 1, 1)
    X_tfm = torch.cat((X, time), 1)
    return X_tfm


def load_pickle(dir):
    with open(dir, "rb") as fp:
        return pickle.load(fp)


def save_pickle(data, dir):
    with open(dir, "wb") as fp:
        pickle.dump(data, fp)


def model_path(config, root="./CIFAR10/models"):

    root = pathlib.Path(root)
    filename = "{}".format(config.dataset)

    # Model-specific keys
    filename += "_model_{}_param_{}_nhid1_{}_nhid2_{}".format(
        config.model,
        config.param,
        config.n_hidden1,
        config.n_hidden2,
    )

    # Optimization arguments
    filename += "_gamma_{}".format(config.gamma)
    filename += "_lr_{}".format(config.lr)

    # Comment
    if config.comment != "":
        filename += "_comment_{}".format(config.comment)

    # Add correct termination
    filename += ".pt"

    # Check if directory exists and warn the user if the it exists and train is used.
    os.makedirs(root, exist_ok=True)
    path = root / filename
    config.path = str(path)

    if config.train and path.exists():
        print("WARNING! The model exists in directory and will be overwritten")


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc == None:
            self.best_acc = val_acc
        elif self.best_acc - val_acc < self.min_delta:
            self.best_acc = val_acc
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_acc - val_acc >= self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
