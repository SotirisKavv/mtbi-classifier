import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedKFold

from utils.plot_fig import plot_accuracies, plot_losses


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

        self.conv11 = nn.Conv2d(1, 128, kernel_size=self.kernel, stride=3, padding=3)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=self.kernel, stride=3, padding=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.dp1 = nn.Dropout(p=0.1)

        self.conv31 = nn.Conv2d(128, 64, kernel_size=self.kernel, stride=3, padding=3)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=self.kernel, stride=3, padding=3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dp2 = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(64, 32 * 1 * 1)
        self.dp3 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # print(f"Input Shape: {x.shape}")
        x = x.unsqueeze(1)
        x = self.dp1(self.pool1(F.relu(self.conv12(F.relu(self.conv11(x))))))
        # print(f"After Conv1 Shape: {x.shape}")
        x = self.dp2(self.pool2(F.relu(self.conv32(F.relu(self.conv31(x))))))
        # print(f"After Conv2 Shape: {x.shape}")
        x = x.view(-1, 64 * 1 * 1)
        # print(f"After View Shape: {x.shape}")
        x = self.dp3(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc3(x))
        # print(f"Output Shape: {x.shape}")
        return x


# Training function
def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    epochs=300,
    patience=15,
    plot=False,
    verbose=True,
):
    model.train()

    best_acc = 0.0
    count_no_improve = 0

    history = []
    for epoch in range(epochs):
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            # print(
            #     f"Inputs Shape: {inputs.shape}, Labels Shape: {labels.shape}, Labels Type: {type(labels)}"
            # )
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"Model Outputs: {outputs}")
            labels = labels.unsqueeze(1)
            # print(
            #     f"Labels Shape Before Loss: {labels.shape}, Labels Type: {type(labels)}"
            # )

            # print(
            #     f"Inputs is None: {inputs is None}, Labels is None: {labels is None}, Outputs is None: {outputs is None}"
            # )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            predicted = (outputs > 0.5).int()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        result = evaluate(model, criterion, valid_loader)
        result["train_loss"] = running_train_loss / len(train_loader)
        result["train_acc"] = correct_train / total_train
        history.append(result)

        if result["val_acc"] > best_acc:
            best_acc = result["val_acc"]
            count_no_improve = 0
        else:
            count_no_improve += 1

        if count_no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

        if verbose:
            print(
                "Epoch [{}]: Training loss: {:.4f} - Training Accuracy: {:.4f} - Validation loss: {:.4f} - Validation accuracy: {:.4f}".format(
                    epoch + 1,
                    result["train_loss"],
                    result["train_acc"],
                    result["val_loss"],
                    result["val_acc"],
                )
            )

    if plot:
        plot_accuracies(history)
        plot_losses(history)
    print("Training complete.")


@torch.no_grad()
def evaluate(model, criterion, valid_loader):
    running_valid_loss = 0.0
    correct_valid = 0
    total_valid = 0

    model.eval()
    for inputs, labels in valid_loader:
        outputs = model(inputs)
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


def predict(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).int()
            predictions.extend(predicted.cpu().numpy().flatten())

    return predictions


def classifyCNN(
    data,
    labels,
    lr=0.001825,
    bsize=3,
    weight_decay=0.0,
    plot=False,
    verbose=True,
    multilayer=False,
):
    model = ClassificationCNN(multilayer)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    data = np.array(data)
    labels = np.array(labels)
    dataset = GraphDataset(data, labels)

    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_size = int(0.8 * train_size)
    valid_size = len(dataset) - train_size - test_size

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=bsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=True)

    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        plot=plot,
        verbose=verbose,
    )

    preds = np.asarray(predict(model, test_loader), dtype=np.int32)

    y_test = np.asarray([label for _, label in test_dataset], dtype=np.int32)
    return y_test, preds


def hyperparameter_tuning(
    data,
    labels,
    learning_rates,
    batch_sizes,
    weight_decays,
    verbose=False,
    multilayer=False,
):
    # Best validation accuracy
    best_val_acc = 0.0

    # Best hyperparameters
    best_lr = learning_rates[0]
    best_bsize = batch_sizes[0]
    best_weight_decay = weight_decays[0]

    for lr in learning_rates:
        for bsize in batch_sizes:
            for weight_decay in weight_decays:
                print(
                    f"Learning Rate: {lr}, Batch Size: {bsize}, Weight Decay: {weight_decay}"
                )
                y_test, preds = classifyCNN(
                    data,
                    labels,
                    lr,
                    bsize,
                    weight_decay,
                    verbose=verbose,
                    multilayer=multilayer,
                )

                acc_mean = (y_test == preds).mean()
                print(f"Accuracy: {acc_mean:.4f}")
                if acc_mean > best_val_acc and acc_mean < 1:
                    best_val_acc = acc_mean
                    best_lr = lr
                    best_bsize = bsize
                    best_weight_decay = weight_decay

    print(f"Best Learning Rate: {best_lr}")
    print(f"Best Batch Size: {best_bsize}")
    print(f"Best Validation Accuracy: {best_val_acc}")
    print(f"Best Weight Decay: {best_weight_decay}")

    return best_lr, best_bsize, best_weight_decay


def cross_validation(dataset, labels, k, lr, bsize, weight_decay, multilayer=False):
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies = []

    dataset = np.array(dataset)
    labels = np.array(labels)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset, labels)):
        print(f"FOLD {fold + 1}")
        X_train, X_val = dataset[train_ids], dataset[val_ids]
        y_train, y_val = labels[train_ids], labels[val_ids]

        # Create GraphDataset for training and validation
        train_dataset = GraphDataset(X_train, y_train)
        val_dataset = GraphDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=True)

        model = ClassificationCNN(multilayer)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_model(model, criterion, optimizer, train_loader, val_loader)

        val_accuracy = evaluate(model, criterion, val_loader)["val_acc"]
        accuracies.append(val_accuracy)

    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"AVERAGE ACCURACY: {average_accuracy:.4f} ± {std_accuracy:.4f}")
    return average_accuracy
