import os
import torch
from datetime import date, datetime

torch.manual_seed(42)
## Move model to cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc * 100


def train(epochs, model, optimizer, criterion, train_loader, test_loader, file_name):
    ## dict should be loaded and not created as new if model is loaded to be trained
    dict_of_results = {
        "train_acc": list(),
        "train_loss": list(),
        "test_acc": list(),
        "test_loss": list(),
        "convergence_time": list(),
    }
    model.to(device)
    with torch.cuda.device(device.index):
        for epoch in range(1, epochs + 1):
            epoch_start_time = datetime.now()
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = (
                    X_train_batch.to(device),
                    y_train_batch.to(device),
                )
                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)

                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = multi_acc(y_train_pred, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            # VALIDATION
            with torch.no_grad():

                test_epoch_loss = 0
                test_epoch_acc = 0

                model.eval()
                for X_test_batch, y_test_batch in test_loader:
                    X_test_batch, y_test_batch = (
                        X_test_batch.to(device),
                        y_test_batch.to(device),
                    )

                    y_test_pred = model(X_test_batch)

                    test_loss = criterion(y_test_pred, y_test_batch)
                    test_acc = multi_acc(y_test_pred, y_test_batch)

                    test_epoch_loss += test_loss.item()
                    test_epoch_acc += test_acc.item()

            epoch_end_time = datetime.now()

            dict_of_results["train_acc"].append(
                round(train_epoch_acc / len(train_loader), 3)
            )
            dict_of_results["train_loss"].append(
                round(train_epoch_loss / len(train_loader), 3)
            )
            dict_of_results["test_acc"].append(
                round(test_epoch_acc / len(test_loader), 3)
            )
            dict_of_results["test_loss"].append(
                round(test_epoch_loss / len(test_loader), 3)
            )
            dict_of_results["convergence_time"].append(
                epoch_end_time - epoch_start_time
            )

            if epoch % 1 == 0:
                print(
                    "Epoch: {} | Train Loss: {} |  Test Loss: {} | Train acc: {} | Test acc: {} | Time taken: {}".format(
                        epoch,
                        round(train_epoch_loss / len(train_loader), 3),
                        round(test_epoch_loss / len(test_loader), 3),
                        round(train_epoch_acc / len(train_loader), 3),
                        round(test_epoch_acc / len(test_loader), 3),
                        epoch_end_time - epoch_start_time,
                    )
                )

            save_results(epoch, model, optimizer, dict_of_results, file_name)


def save_results(epoch, model, optimizer, dict_of_results, file_name):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "results": dict_of_results,
    }

    # Check if folder structure is created, if not - create it
    if not os.path.isdir("Results"):
        os.makedirs("Results")
    torch.save(state, os.path.join("Results", file_name + ".pth"))
