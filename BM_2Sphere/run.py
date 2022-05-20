import torch
import os
import numpy as np
import copy
#from src.runner.test import test
import datetime
import ml_collections
import yaml
from BM_2Sphere.utils import model_path
import torch.nn.functional as F
import argparse


def _train_BM_2Sphere(
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
    criterion = torch.nn.MSELoss()
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
        # log learning_rate of the epoch

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            total = 0
            # iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
               # print('input shape=', inputs.shape)
                labels = labels.to(device)
                #print('lable shape=', labels.shape)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)
                 #   print('output shape =', outputs.shape)
                    loss = criterion(outputs, labels)
                    # Regularization:
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            print("{} Loss: {:.4f}".format(
                phase, epoch_loss))
            print(datetime.datetime.now())
            if phase == "validation":

                val_loss.append(epoch_loss)

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_loss <= best_loss:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_loss == best_loss):
                    pass
                else:
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results

                    test_loss = _test_BM_2Sphere(model, test_loader, config)
        if counter > config.patience:
            break
        else:
            lr_scheduler.step()
            print()

        lr_scheduler.step()
        print()
    # Report best results
    #print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), config.path)
    # Return model and histories
    return model


def _test_BM_2Sphere(model, test_loader, config):
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():
        # Iterate through data
        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)
            # lengths = lengths
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

    # Print results
    test_loss = running_loss / total
    print(
        "MSE of the network on the {} test samples: {}".format(
            total, (test_loss)
        )
    )
    return test_loss


def main(config):
    # print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # Set the seed
    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    # initialize weight and bias
    # Place here your API key.

    if (config.device ==
            "cuda" and torch.cuda.is_available()):
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    from BM_2Sphere.models import get_model
    model = get_model(config)
    from BM_2Sphere.dataloader import get_dataset
    # Define transforms and create dataloaders
    dataloaders, test_loader = get_dataset(config, num_workers=4)

    # Create model directory and instantiate config.path
    model_path(config)

    if config.pretrained:
        # Load model state dict
        model.module.load_state_dict(torch.load(config.path), strict=False)

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        # Train the model
        import datetime

        print(datetime.datetime.now())
        _ = _train_BM_2Sphere(
            model, dataloaders, config, test_loader)

    # Select test function
    test_loss = _test_BM_2Sphere(model, test_loader, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM_DEV',
                        help='choose from LSTM, LSTM_DEV,EXPRNN')
    args = parser.parse_args()
    if args.model == 'LSTM_DEV':
        with open('BM_2Sphere/configs/train_lstm_dev.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    elif args.model == 'LSTM':
        with open('BM_2Sphere/configs/train_lstm.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    elif args.model == 'EXPRNN':
        with open('BM_2Sphere/configs/train_exprnn.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
    main(config=config)
