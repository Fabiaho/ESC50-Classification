import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
import sys
from functools import partial

from models.model_classifier import ResNet, ResidualBlock, GetACDNetModel
from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50
import config

import warnings

warnings.filterwarnings("ignore")


# evaluate model on different testing data 'dataloader'
def test(model, dataloader, criterion, device):
    model.eval()

    y_pred = None
    y_target = None
    y_probs = {}
    running_loss = 0.0
    file_names = []
    
    with torch.no_grad():
        for k, x, y in tqdm(dataloader, unit="bat", disable=config.disable_bat_pbar, position=0):
            # Preprocess the input data similarly to train_epoch
            x = np.asarray(x)
            y = np.asarray(y)

            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])
            
            x = np.expand_dims(x, axis=1)
            x = np.expand_dims(x, axis=3)
            
            x = torch.tensor(np.moveaxis(x, 3, 1)).to(device)
            y = torch.tensor(y).to(device)

            # The forward pass through the model
            scores = model(x)
            
            loss = criterion(scores.log(), y)
            running_loss += loss.item()

            y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data))
            y_target = y if y_target is None else torch.cat((y_target, y))
            
            file_names.extend(k)
            
        
        # Reshape to incorporate crops
        y_pred = y_pred.reshape(y_pred.shape[0] // config.nCrops, config.nCrops, y_pred.shape[1])
        y_target = y_target.reshape(y_target.shape[0] // config.nCrops, config.nCrops, y_target.shape[1])
        
        # Average over the crops
        y_pred_avg = y_pred.mean(dim=1)
        y_target_avg = y_target.mean(dim=1)
        
        y_pred = y_pred_avg.argmax(dim=1)
        y_target = y_target_avg.argmax(dim=1)
        
        # Calculate accuracy
        acc = (((y_pred == y_target) * 1).float().mean() * 100).item()
        
        loss = running_loss / (y_target.shape[0] * config.nCrops)
        
        for file_name in file_names:
            y_probs[file_name] = y_pred_avg[file_names.index(file_name)].cpu().numpy()
            
    return acc, loss, y_probs

def get_lr(epoch):
        divide_epoch = np.array([config.epochs * i for i in config.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= config.warm_epochs:
            decay = 1;
        return config.lr * np.power(0.1, decay);

def train_epoch(epoch):
    # switch to training
    model.train()
    optimizer.param_groups[0]['lr'] = get_lr(epoch);
    losses = []
    running_loss = 0.0;
    total_correct = 0;
    samples_count = 0
    for _, x, label in tqdm(
        train_loader, unit="bat", disable=config.disable_bat_pbar, position=0
    ):
        # x = x.float().to(device)
        # y_true = label.to(device)
        x = np.expand_dims(x, axis=1);
        x = np.expand_dims(x, axis=3);
        x = torch.tensor(np.moveaxis(x, 3, 1)).to(device)
        y = torch.tensor(label).to(device)

        # the forward pass through the model
        outputs = model(x)

        correct_predictions = (outputs.data.argmax(dim=1) == y.argmax(dim=1)).sum().item()
        loss = criterion(outputs.log(), y)
        # reset the gradients to zero - avoids accumulation
        optimizer.zero_grad()
        # compute the gradient with backpropagation
        loss.backward()
        losses.append(loss.item())
        # minimize the loss via the gradient - adapts the model parameters
        optimizer.step()

        running_loss += loss.item();
        total_correct += correct_predictions
        samples_count += outputs.shape[0]

    acc = (total_correct / samples_count)*100;
    losses = running_loss / samples_count;
    return acc, losses


def fit_classifier():
    num_epochs = config.epochs

    loss_stopping = EarlyStopping(
        patience=config.patience,
        #higher_better=True,
        delta=0.002,
        verbose=True,
        float_fmt=float_fmt,
        checkpoint_file=os.path.join(experiment, "best_val_acc.pt"),
    )

    pbar = tqdm(
        range(1, 1 + num_epochs), ncols=50, unit="ep", file=sys.stdout, ascii=True
    )
    for epoch in range(1, 1 + num_epochs):
        # iterate once over training data
        train_acc, train_loss = train_epoch(epoch)

        """# validate model
        val_acc, val_loss, _ = test(
            model, val_loader, criterion=criterion, device=device
        )
        val_loss_avg = np.mean(val_loss)"""

        # print('\n')
        pbar.update()
        # pbar.refresh() syncs output when pbar on stderr
        # pbar.refresh()
        print(end=" ")
        print(  # f" Epoch: {epoch}/{num_epochs}",
            f"TrnAcc={train_acc:{float_fmt}}",
            f"TrnLoss={np.mean(train_loss):{float_fmt}}",
            #f"ValAcc={val_acc:{float_fmt}}",
            #f"ValLoss={val_loss_avg:{float_fmt}}",
            end=" ",
        )
        
        print()

        """early_stop, improved = loss_stopping(val_loss_avg, model, epoch)
        if not improved:
            print()
        if early_stop:
            print("Early stopping")
            break"""

        # advance the optimization scheduler
        # scheduler.step()
    # save full model
    torch.save(model.state_dict(), os.path.join(experiment, "terminal.pt"))


# build model from configuration.
def make_model():
    n = config.n_classes
    model_constructor = config.model_constructor
    print(model_constructor)
    model = eval(model_constructor)
    return model


if __name__ == "__main__":
    data_path = config.esc50_path
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{config.device_id}" if use_cuda else "cpu")

    # digits for logging
    float_fmt = ".3f"
    pd.options.display.float_format = ("{:," + float_fmt + "}").format
    runs_path = config.runs_path
    experiment_root = os.path.join(
        runs_path, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    )
    if not os.path.exists(experiment_root):
        os.mkdir(experiment_root)

    # for all folds
    scores = {}
    for test_fold in config.test_folds:
        experiment = os.path.join(experiment_root, f"{test_fold}")
        if not os.path.exists(experiment):
            os.mkdir(experiment)

        # clone stdout to file (does not include stderr). If used may confuse linux 'tee' command.
        with Tee(
            os.path.join(experiment, "train.log"),
            "w",
            1,
            encoding="utf-8",
            newline="\n",
            proc_cr=True,
        ):
            # this function assures consistent 'test_folds' setting for train, val, test splits
            get_fold_dataset = partial(
                ESC50, test_folds={test_fold}, root=data_path, download=True
            )

            train_set = get_fold_dataset(subset="train")
            print("*****")
            print(
                f"train folds are {train_set.train_folds} and test fold is {train_set.test_folds}"
            )
            print("random wave cropping")

            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                drop_last=False,
                persistent_workers=config.persistent_workers,
                # pin_memory=True,
            )

            val_loader = torch.utils.data.DataLoader(
                get_fold_dataset(subset="val"),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                drop_last=False,
                persistent_workers=config.persistent_workers,
            )

            print()
            # instantiate model
            model = make_model()
            # model = nn.DataParallel(model, device_ids=config.device_ids)
            model = model.to(device)
            print("*****")

            # Define a loss function and optimizer
            criterion = nn.KLDivLoss(reduction="batchmean")

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                nesterov=True,
            )

            """scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.step_size, gamma=config.gamma
            )"""

            # fit the model using only training and validation data, no testing data allowed here
            print()
            fit_classifier()

            # tests
            test_loader = torch.utils.data.DataLoader(
                get_fold_dataset(subset="test"),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,  # config.num_workers,
                drop_last=False,
            )

            print(f"\ntest {experiment}")
            test_acc, test_loss, _ = test(
                model, test_loader, criterion=criterion, device=device
            )
            scores[test_fold] = pd.Series(
                dict(TestAcc=test_acc, TestLoss=np.mean(test_loss))
            )
            print(scores[test_fold])
            # print(scores[test_fold].unstack())
            print()
    scores = pd.concat(scores).unstack([-1])
    print(pd.concat((scores, scores.agg(["mean", "std"]))))
