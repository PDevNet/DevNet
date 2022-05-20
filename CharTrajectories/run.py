
import torch
import os
import numpy as np
import copy
import datetime
import ml_collections
import yaml
from CharTrajectories.utils import EarlyStopping, model_path
import argparse


def train_CT(
    model, dataloader, config, test_loader,
):

    # Training parameters
    epochs = config.epochs
    device = config.device
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
    # wandb.watch(model, criterion, log="all", log_freq=1)
    counter = 0
    val_acc = []
    val_loss = []
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
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
            if phase == "validation":
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log best results so far and the weights of the model.

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results

                    test_acc = test_CT(model, test_loader, config)

            if phase == "validation":
                print('validation and schedulaer')
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'max').step(metrics=epoch_acc)
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
    # Save the model in the exchangeable ONNX format
    torch.save(model.state_dict(), config.path)
    # Return model and histories
    return model, val_loss, val_acc


def test_CT(model, test_loader, config):
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0

    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:

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


def main(**kwargs):
    if "config" in kwargs:
        config = kwargs["config"]
    else:
        with open('CharTrajectories/sweep.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    if "return_loss" in kwargs:
        return_loss = kwargs['return_loss']
    else:
        return_loss = False
    # print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Set the seed
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    print(config)
    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    from CharTrajectories.models import get_model
    model = get_model(config)

    # Define transforms and create dataloaders
    from CharTrajectories.dataloader import get_dataset
    dataloaders, test_loader = get_dataset(config, num_workers=4)

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
        _, val_loss, val_acc = train_CT(
            model, dataloaders, config, test_loader)

    # Select test function
    test_acc = test_CT(model, test_loader, config)

    if return_loss:
        return val_loss, val_acc
    else:
        return test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM_DEV',
                        help='choose from LSTM, DEV, LSTM_DEV, signature,EXPRNN')

    parser.add_argument('--param', type=str, default='SO',
                        help='choose from SO,Sp')
    parser.add_argument('--drop_rate', type=float, default=0.3,
                        help='drop rate choosen from 0.3,0.5,0.7')
    parser.add_argument('--train_sr', type=float, default=1,
                        help='train sampling rate from 1,0.5')
    parser.add_argument('--test_sr', type=float, default=1,
                        help='train sampling rate from 1,0.5')

    args = parser.parse_args()
    if args.model == 'LSTM_DEV':
        with open('CharTrajectories/configs/train_lstm_dev.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
            config.param = args.param
    elif args.model == 'LSTM':
        with open('CharTrajectories/configs/train_lstm.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    elif args.model == 'DEV':
        with open('CharTrajectories/configs/train_dev.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
            config.param = args.param
    elif args.model == 'signature':
        with open('CharTrajectories/configs/train_sig.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    elif args.model == 'EXPRNN':
        with open('CharTrajectories/configs/train_exprnn.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    print(args)
    config.drop_rate = args.drop_rate
    config.train_sr = args.train_sr
    config.test_sr = args.test_sr
    main(config=config)
