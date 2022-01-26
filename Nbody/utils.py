import os
import pathlib
import torch
import ml_collections
import yaml


def model_path(config, root="./Nbody/models"):

    root = pathlib.Path(root)
    filename = "{}".format(config.dataset)

    # Model-specific keys
    filename += "_model_{}_param_{}_nhid1_{}_nhid2_{}_p_{}".format(
        config.model,
        config.param,
        config.n_hid1,
        config.n_hid2,
        config.p,
    )

    # Optimization arguments
    if config.scheduler == "plateau":
        filename += "_pat_{}".format(config.sched_patience)
    elif config.scheduler == 'multistep':
        filename += "_schsteps_{}".format(config.sched_decay_steps)
    elif config.scheduler == 'exponential':
        filename += "_gamma_{}".format(config.gamma)

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
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def count_parameters(model: torch.nn.Module) -> int:
    """

    Args:
        model (torch.nn.Module): input models
    Returns:
        int: number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(file_dir: str):
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config
