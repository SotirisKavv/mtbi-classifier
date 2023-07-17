import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from src.utils.plot_fig import plot_accuracies, plot_losses


class ClassificationBase(nn.Module):
    def training_step(self, batch):
        graphs, labels = zip(*batch)
        out = self(graphs)  # Generate predictions
        loss = nn.BCELoss(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        graphs, labels = zip(*batch)
        out = self(graphs)  # Generate predictions
        loss = nn.BCELoss(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


class ClassificationCNN(ClassificationBase):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 11 * 11, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.conv12(F.relu(self.conv11(x)))))
        x = self.pool2(F.relu(self.conv22(F.relu(self.conv21(x)))))
        x = self.pool3(F.relu(self.conv32(F.relu(self.conv31(x)))))

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, bsize, model, train_data, val_data):
    opt_func = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        train_losses = []
        i = 0
        while i < len(train_data):
            loss = model.training_step(train_data[i : min(i + bsize, len(train_data))])
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += bsize
        # Validation phase
        result = evaluate(model, val_data)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    plot_accuracies(history)
    plot_losses(history)


def predict(model, test_data):
    model.eval()
    with torch.no_grad():
        graphs, _ = zip(*test_data)
        out = model(graphs)
        preds = (out > 0.5).float()
    return preds.tolist()


def classifyCNN(data):
    model = ClassificationCNN()
    model.load_state_dict(torch.load("../../models/classificationCNN.pth"))

    X_train, X_test, y_train, y_test = train_test_split(
        data["graphs"], data["labels"], test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    fit(10, 0.001, 3, model, list(zip(X_train, y_train)), list(zip(X_val, y_val)))

    preds = predict(model, list(zip(X_test, y_test)))

    return y_test, preds
