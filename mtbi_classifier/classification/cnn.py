import torch
import numpy as np
from tabulate import tabulate
from hyperopt import fmin, tpe, hp, STATUS_OK
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    auc,
    roc_curve,
)


class GraphDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ClassificationCNN(nn.Module):
    def __init__(self, multilayer=False):
        super(ClassificationCNN, self).__init__()
        self.kernel = 7 if multilayer else 3

        #   1st Convolutional Layer
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module(
            "Conv11", nn.Conv2d(1, 128, kernel_size=self.kernel, stride=3, padding=3)
        )
        self.layer_1.add_module("BN11", nn.BatchNorm2d(128))
        self.layer_1.add_module("ReLU11", nn.ReLU(inplace=False))
        self.layer_1.add_module(
            "Conv12", nn.Conv2d(128, 128, kernel_size=self.kernel, stride=3, padding=3)
        )
        self.layer_1.add_module("BN12", nn.BatchNorm2d(128))
        self.layer_1.add_module("ReLU12", nn.ReLU(inplace=False))
        self.layer_1.add_module("MaxPool1", nn.MaxPool2d(2, 2))
        self.layer_1.add_module("Dropout1", nn.Dropout(p=0.1))

        #   2nd Convolutional Layer
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module(
            "Conv21", nn.Conv2d(128, 64, kernel_size=self.kernel, stride=3, padding=3)
        )
        self.layer_2.add_module("BN21", nn.BatchNorm2d(64))
        self.layer_2.add_module("ReLU21", nn.ReLU(inplace=False))
        self.layer_2.add_module(
            "Conv22", nn.Conv2d(64, 64, kernel_size=self.kernel, stride=3, padding=3)
        )
        self.layer_2.add_module("BN22", nn.BatchNorm2d(64))
        self.layer_2.add_module("ReLU22", nn.ReLU(inplace=False))
        self.layer_2.add_module("MaxPool2", nn.MaxPool2d(2, 2))
        self.layer_2.add_module("Dropout2", nn.Dropout(p=0.1))

        #   Fully Connected Layer
        self.fully_connected = nn.Sequential()
        self.fully_connected.add_module("Linear1", nn.Linear(64, 32 * 1 * 1))
        self.fully_connected.add_module("Dropout1", nn.Dropout(p=0.1))
        self.fully_connected.add_module("Linear2", nn.Linear(32, 1))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.view(-1, 64 * 1 * 1)
        x = self.fully_connected(x)
        x = torch.sigmoid(x)
        return x


class CNNClassifier:
    def __init__(
        self,
        learning_rate=0.001825,
        weight_decay=0.00,
        batch_size=5,
        gamma=0.9,
        multilayer=False,
    ):
        self.lr = learning_rate
        self.wd = weight_decay
        self.bs = batch_size
        self.gamma = gamma
        self.cnn = ClassificationCNN(multilayer)

    def train(
        self,
        criterion,
        optimizer,
        train_loader,
        valid_loader,
        epochs=300,
        patience=15,
        verbose=True,
    ):
        self.cnn.train()

        best_val_loss = float("inf")
        count_no_improve = 0
        scheduler = StepLR(optimizer, step_size=10, gamma=self.gamma)

        history = []
        for epoch in range(epochs):
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.cnn(inputs)
                labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                predicted = (outputs > 0.5).int()
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            scheduler.step()
            result = self.evaluate(criterion, valid_loader)
            result["train_loss"] = running_train_loss / len(train_loader)
            result["train_acc"] = correct_train / total_train
            history.append(result)

            if result["val_loss"] < best_val_loss:
                best_val_loss = result["val_loss"]
                count_no_improve = 0
            else:
                count_no_improve += 1
                if count_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose:
                print(
                    "Epoch [{}]:\tTraining loss: {:.4f}\tTraining Accuracy: {:.4f}\tValidation loss: {:.4f}\tValidation accuracy: {:.4f}".format(
                        epoch + 1,
                        result["train_loss"],
                        result["train_acc"],
                        result["val_loss"],
                        result["val_acc"],
                    )
                )
        return history

    @torch.no_grad()
    def evaluate(self, criterion, valid_loader):
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        self.cnn.eval()
        for inputs, labels in valid_loader:
            outputs = self.cnn(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            running_valid_loss += loss.item()
            predicted = (outputs > 0.5).int()
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()

        return {
            "val_loss": running_valid_loss / len(valid_loader),
            "val_acc": correct_valid / total_valid,
        }

    def predict(self, test_loader):
        self.cnn.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.cnn(inputs)
                predicted = (outputs > 0.5).int()
                true_labels.extend(labels.tolist())
                predictions.extend(predicted.cpu().numpy().flatten())

        return np.asarray(true_labels, dtype=np.int32), np.asarray(
            predictions, dtype=np.int32
        )

    def classify(
        self,
        data,
        labels,
        verbose=True,
        multilayer=False,
    ):
        self.cnn = ClassificationCNN(multilayer)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.cnn.parameters(), lr=self.lr, weight_decay=self.wd)
        data = np.array(data)
        labels = np.array(labels)
        dataset = GraphDataset(data, labels)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_size = int(0.75 * train_size)
        valid_size = len(dataset) - train_size - test_size

        train_dataset, valid_dataset = random_split(
            train_dataset, [train_size, valid_size]
        )

        bsize = self.bs
        train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=bsize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=True)

        self.train(
            criterion,
            optimizer,
            train_loader,
            val_loader,
            verbose=verbose,
        )

        y_lab, preds = self.predict(test_loader)
        return y_lab, preds

    def hyperparameter_tuning(self, data, labels, multilayer=False, max_evals=25):
        data = np.array(data)
        labels = np.array(labels)

        def objective(params):
            skf = StratifiedKFold(n_splits=5)
            val_accuracies = []

            for train_idx, val_idx in skf.split(data, labels):
                self.cnn = ClassificationCNN(multilayer)

                params["batch_size"] = int(params["batch_size"])
                self.lr = params["lr"]
                self.bs = params["batch_size"]
                self.wd = params["weight_decay"]
                self.gamma = params["gamma"]

                X_train, X_val = data[train_idx], data[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                train_dataset = GraphDataset(X_train, y_train)
                val_dataset = GraphDataset(X_val, y_val)
                train_loader = DataLoader(
                    train_dataset, batch_size=self.bs, shuffle=True
                )
                val_loader = DataLoader(val_dataset, batch_size=self.bs, shuffle=False)

                criterion = nn.BCELoss()
                optimizer = optim.Adam(
                    self.cnn.parameters(), lr=self.lr, weight_decay=self.wd
                )
                history = self.train(
                    criterion, optimizer, train_loader, val_loader, verbose=False
                )
                val_accuracies.append(history[-1]["val_acc"])

            # Average validation accuracy across all folds
            avg_val_acc = np.mean(val_accuracies)
            return {"loss": -avg_val_acc, "status": STATUS_OK}

        # Define the search space
        space = {
            "lr": hp.loguniform("lr", -10, -4),
            "batch_size": hp.quniform("batch_size", 3, 6, 1),
            "weight_decay": hp.loguniform("weight_decay", -10, -4),
            "gamma": hp.uniform("gamma", 0.8, 1),
        }

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            verbose=2,
        )

        best["batch_size"] = int(best["batch_size"])

        self.lr = best["lr"]
        self.bs = best["batch_size"]
        self.wd = best["weight_decay"]
        self.gamma = best["gamma"]

        optimal_hyperparameters = {
            "Learning Rate": [best["lr"]],
            "Batch Size": [best["batch_size"]],
            "Weight Decay": [best["weight_decay"]],
            "Gamma": [best["gamma"]],
        }

        print(tabulate(optimal_hyperparameters, headers="keys", tablefmt="fancy_grid"))

    def cross_validate(self, dataset, labels, cv, multilayer=False, verbose=False):
        kfold = StratifiedKFold(n_splits=cv, shuffle=True)
        accuracies = []

        tprs = []
        mean_fpr = np.linspace(0, 1, 1000)

        histories = []

        targets = []
        predictions = []

        dataset = np.array(dataset)
        labels = np.array(labels)

        for train_ids, val_ids in kfold.split(dataset, labels):
            self.cnn = ClassificationCNN(multilayer)

            X_train, X_val = dataset[train_ids], dataset[val_ids]
            y_train, y_val = labels[train_ids], labels[val_ids]

            # Create GraphDataset for training and validation
            train_dataset = GraphDataset(X_train, y_train)
            val_dataset = GraphDataset(X_val, y_val)

            bsize = self.bs
            train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=bsize)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                self.cnn.parameters(), lr=self.lr, weight_decay=self.wd
            )

            h = self.train(
                criterion, optimizer, train_loader, val_loader, verbose=verbose
            )
            y_lab, preds = self.predict(val_loader)
            print(f"targets: {y_lab}\tpredicted: {preds}", end="\r")
            targets.extend(y_val)
            predictions.extend(preds)

            accuracies.append(accuracy_score(y_val, preds))

            fpr, tpr, _ = roc_curve(y_val, preds)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            histories.append(h)

        print(end="\x1b[2K")
        print(classification_report(targets, predictions))

        # Accuracy Statistics
        accuracies = np.array(accuracies)
        acc_params = accuracies.mean(), accuracies.std()

        # ROC Statistics
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        auc_roc = auc(mean_fpr, mean_tpr)
        roc_params = mean_fpr, mean_tpr, std_tpr, auc_roc

        history = histories[accuracies.argmax()]

        return acc_params, roc_params, history
