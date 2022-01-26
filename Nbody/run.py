import torch.nn.functional as F
from utils import model_path
from dataloader import get_dataset
import yaml
import ml_collections
import os
import torch
import copy
import datetime
import argparse


def train_Nbody(model, dataloader, config, test_loader
                ):

    # Training parameters
    epochs = config.epochs
    device = config.device

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())

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
    # wandb.watch(model, criterion, log="all", log_freq=1)
    counter = 0
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
            bs_running_loss = 0

            total = 0
            # iterate over data
            for inputs, targets in dataloader[phase]:

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    # inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                bs_running_loss += F.mse_loss(inputs[:, -1, :, :2],
                                              targets)*inputs.size(0)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
            bs_loss = bs_running_loss / total
            # statistics of the epoch
            epoch_loss = running_loss / total

            print("{} Loss: {:.4f}".format(
                phase, epoch_loss))
            print(datetime.datetime.now())

            # log statistics of the epoch
            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_loss <= best_loss:
                counter = 0
                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_loss == best_loss):
                    pass
                else:
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log best results so far and the weights of the model.

                    # Clean CUDA Memory
                    del inputs, outputs
                    torch.cuda.empty_cache()
                    # Perform test and log results
                    test_loss, bs_loss = test_Nbody(
                        model, test_loader, config)

            elif phase == "validation" and epoch_loss > best_loss:
                counter += 1
        if counter > config.patience:
            break
        else:
            lr_scheduler.step()
            print()
    # Report best results
    print("Best Val Loss: {:.4f}".format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), config.path)
    # Save the model in the exchangeable ONNX format
    torch.save(model.state_dict(), config.path)
    # Return model and histories
    return model


def test_Nbody(model, test_loader, config):
    # send model to device
    device = config.device

    model.eval()
    model.to(device)

    # Summarize results

    running_loss = 0

    total = 0
    bs_running_loss = 0

    with torch.no_grad():
        # Iterate through data
        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            total += inputs.size(0)
            bs_running_loss += F.mse_loss(inputs[:, -1, :, :2],
                                          targets)*inputs.size(0)

    total_loss = running_loss/total

    bs_loss = bs_running_loss/total

    # Print results
    print("MSE of the network on the {}, test samples pos loss: {}, baseline_pos_loss: {}".format(
        total, (total_loss), (bs_loss)
    )
    )
    return total_loss, bs_loss


def main(**kwargs):
    if "config" in kwargs:
        config = kwargs["config"]
    else:
        with open('Nbody/train.yaml') as file:
            config = ml_collections.ConfigDict(yaml.safe_load(file))
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
    from models import get_model
    model = get_model(config)

    # Define transforms and create dataloaders
    dataloaders, test_loader = get_dataset(config)

    # Create model directory and instantiate config.path
    model_path(config)

    if config.pretrained:
        # Load model state dict
        model.module.load_state_dict(torch.load(config.path), strict=False)

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        # Train the model

        train_Nbody(model, dataloaders, config, test_loader)

    # Select test function
    test_acc = test_Nbody(model, test_loader, config)[0]
    return test_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run program')
    parser.add_argument('--gpu_id', type=str, default='0')
    # LSTM, LSMT_development
    parser.add_argument('--model', type=str, default='LSTM_DEV')
    parser.add_argument('--param', type=str, default='SE')  # SO
    parser.add_argument('--p', type=int, default=30)  # 10,30,50
    args = parser.parse_args()
    print(args)
    with open('Nbody/train.yaml') as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    config.gpu_id = args.gpu_id
    config.model = args.model
    config.param = args.param
    config.p = args.p

    main(config=config)
