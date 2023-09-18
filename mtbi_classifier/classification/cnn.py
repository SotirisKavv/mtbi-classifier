import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from utils.plot_fig import plot_accuracies, plot_losses


#   Dataset definition
class GraphDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        graphs = self.X[idx]
        target = self.y[idx]
        return graphs, target


# Define the Convolutional Neural Network
class ClassificationCNN(nn.Module):
    def __init__(self):
        super(ClassificationCNN, self).__init__()

        self.conv11 = nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=3)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.dp1 = nn.Dropout(p=0.5)
        # self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(2, 2)

        self.conv31 = nn.Conv2d(64, 512, kernel_size=3, stride=3, padding=3)
        self.conv32 = nn.Conv2d(512, 512, kernel_size=3, stride=3, padding=3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.dp2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(512 * 1 * 1, 256)
        self.dp3 = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dp1(self.pool1(F.relu(self.conv12(F.relu(self.conv11(x))))))
        # x = self.pool2(F.relu(self.conv22(F.relu(self.conv21(x)))))
        x = self.dp2(self.pool3(F.relu(self.conv32(F.relu(self.conv31(x))))))

        # print(x.shape)

        x = x.view(-1, 512 * 1 * 1)
        # x = F.relu(self.fc2(x))
        x = self.dp3(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc3(x))
        return x


# Training function
def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs=10):
    model.train()

    history = []
    for epoch in range(epochs):
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            # print(outputs.squeeze(), labels.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_train_loss += loss.item()
            predicted = (outputs > 0.5).int()
            # print(predicted.squeeze(), labels.squeeze())
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Validation
        result = evaluate(model, criterion, valid_loader)
        result["train_loss"] = running_train_loss / len(train_loader)
        result["train_acc"] = correct_train / total_train
        history.append(result)
        print(
            "Epoch [{}]: Training loss: {:.4f} - Training Accuracy: {:.4f} - Validation loss: {:.4f} - Validation accuracy: {:.4f}".format(
                epoch + 1,
                result["train_loss"],
                result["train_acc"],
                result["val_loss"],
                result["val_acc"],
            )
        )

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
        print(outputs.squeeze(), labels.squeeze())
        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        running_valid_loss += loss.item()
        predicted = (outputs > 0.5).int()
        # print(predicted.squeeze(), labels.squeeze())
        total_valid += labels.size(0)
        correct_valid += (predicted == labels).sum().item()

    return {
        "val_loss": running_valid_loss / len(valid_loader),
        "val_acc": correct_valid / total_valid,
    }


# Prediction function
def predict(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).int()
            print("Target: {} - Predicted: {}".format(labels, predicted.squeeze()))
            predictions.extend(predicted.cpu().numpy().flatten())

    return predictions


def classifyCNN(data, labels):
    lr, bsize, epochs = 0.00182, 3, 25
    model = ClassificationCNN()
    # criterion = F.binary_cross_entropy
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = GraphDataset(data, labels)

    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_size = int(0.8 * train_size)
    valid_size = len(dataset) - train_size - test_size

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    print(train_size, valid_size, test_size, len(dataset))

    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=bsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=True)

    train_model(model, criterion, optimizer, train_loader, val_loader, epochs)
    preds = np.asarray(predict(model, test_loader), dtype=np.int32)

    y_test = np.asarray([label for _, label in test_dataset], dtype=np.int32)
    return y_test, preds
