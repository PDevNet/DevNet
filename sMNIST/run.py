from sMNIST.utils import count_parameters
import torch
import os
import numpy as np
import copy
#from src.runner.test import test
import datetime
import ml_collections
import yaml
from sMNIST.models import get_model
from sMNIST.dataloader import get_dataset
from sMNIST.utils import model_path, EarlyStopping
import argparse


def train_Mnist(
    model, dataloader, config, test_loader
):
    permutation = torch.Tensor(
        np.random.permutation(784).astype(np.float64)).long()
    # Training parameters
    epochs = config.epochs
    device = config.device
    # clip = config.clip

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999
    # iterate over epochs
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=config.gamma)
    criterion = torch.nn.CrossEntropyLoss()
    counter = 0
    # wandb.watch(model, criterion, log="all", log_freq=1)
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:

                _, in_channels, x, y = inputs.shape
                inputs = inputs.view(-1, in_channels, x * y)
                if config.permuted and config.dataset == "MNIST":
                    inputs = inputs[:, :, permutation]
                inputs = inputs.permute(0, 2, 1).to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Regularization:
                    _, preds = torch.max(outputs, 1)
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))
            print(datetime.datetime.now())

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results

                    test_acc = best_acc
            if phase == "validation":
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min').step(metrics=best_loss)
                EarlyStopping(patience=30)(val_acc=best_acc)
        if counter > config.patience:
            break
        else:
            lr_scheduler.step()
            print()

        lr_scheduler.step()
        print()
    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), config.path)
    # Return model and histories
    return model


def test_Mnist(model, test_loader, config):
    # send model to device
    permutation = torch.Tensor(
        np.random.permutation(784).astype(np.float64)).long()
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0

    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:
            _, in_channels, x, y = inputs.shape
            inputs = inputs.view(-1, in_channels, x * y)
            inputs = inputs[:, :, permutation]

            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    test_acc = correct / total
    print(
        "Accuracy of the network on the {} test samples: {}".format(
            total, (100 * test_acc)
        )
    )
    return test_acc


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Set the seed
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    # initialize weight and bias
    # Place here your API key.
    os.environ["WANDB_API_KEY"] = "0a2ae01d4ea2b07b7fca1f71e45562ab1a123c80"
    if not config.train:
        os.environ["WANDB_MODE"] = "dryrun"

    print(config)
    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)
    torch.cuda.set_per_process_memory_fraction(0.5, 0)

    from sMNIST.models import get_model
    model = get_model(config)
    num_param = count_parameters(model)
    print('num_param;', num_param)

    # Define transforms and create dataloaders
    dataloaders, test_loader = get_dataset(config, num_workers=4)

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    # wandb.watch(model, log="all", log_freq=200) # -> There was a wandb bug that made runs in Sweeps crash

    # Create model directory and instantiate config.path
    model_path(config)

    if config.pretrained:
        # Load model state dict
        model.module.load_state_dict(torch.load(config.path), strict=False)

    # Train the model
    if config.train:
        # Train the model
        import datetime

        print(datetime.datetime.now())
        train_Mnist(model, dataloaders, config, test_loader)

    # Select test function
    test_acc = test_Mnist(model, test_loader, config)
    return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM_DEV',
                        help='choose from LSTM, LSTM_DEV')
    parser.add_argument('--permuted', type=str, default='True',
                        help='choose from True, False'
                        )
    args = parser.parse_args()
    if args.model == 'LSTM_DEV':
        with open('sMNIST/configs/train_lstm_dev.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    elif args.model == 'LSTM':
        with open('sMNIST/configs/train_lstm.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))

    if args.permuted == 'True':
        config.permuted = True
    elif args.permute == 'False':
        config.permuted = False
    main(config)
