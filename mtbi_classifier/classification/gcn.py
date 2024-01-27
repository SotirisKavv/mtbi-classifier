import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GCNConv, global_max_pool, BatchNorm
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    classification_report,
)
from hyperopt import fmin, tpe, hp, STATUS_OK
from tabulate import tabulate


def convert_to_pyg_format(feature_matrix, adj_matrix, label):
    edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
    x = torch.tensor(
        feature_matrix
        if adj_matrix.shape[0] == feature_matrix.shape[0]
        else feature_matrix.T,
        dtype=torch.float,
    )
    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


class GraphDataset(Dataset):
    def __init__(self, feature_matrices, adj_matrices, labels):
        super(GraphDataset, self).__init__()
        self.data_list = [
            convert_to_pyg_format(fm, am, l)
            for fm, am, l in zip(feature_matrices, adj_matrices, labels)
        ]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class DropEdge(nn.Module):
    def __init__(self, drop_prob):
        super(DropEdge, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, edge_index):
        mask = torch.rand(edge_index.size(1)) > self.drop_prob
        edge_index = edge_index[:, mask]
        return edge_index


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(9, 512)
        self.conv2 = GCNConv(512, 512)
        self.batch_norm1 = BatchNorm(9)  # Normalizing initial 9 features
        self.batch_norm2 = BatchNorm(512)  # Normalizing 512 features after conv1
        self.batch_norm3 = BatchNorm(512)  # Normalizing 512 features after conv2
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.drop_edge = DropEdge(drop_prob=0.2)
        self.dropout = nn.Dropout(0.9)

    def forward(self, x, edge_index):
        x = self.batch_norm1(x)
        edge_index = self.drop_edge(edge_index)
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm2(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.batch_norm3(x)
        x = global_max_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class GCNClassifier:
    def __init__(
        self,
        learning_rate=0.001825,
        weight_decay=0.00,
        gamma=0.9,
    ):
        self.lr = learning_rate
        self.wd = weight_decay
        self.gamma = gamma
        self.gcn = GCN()

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
        self.gcn.train()

        best_val_loss = float("inf")
        count_no_improve = 0
        scheduler = StepLR(optimizer, step_size=10, gamma=self.gamma)

        history = []
        for epoch in range(epochs):
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for data in train_loader:
                optimizer.zero_grad()
                outputs = self.gcn(data.x, data.edge_index)
                labels = data.y.unsqueeze(1)
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

        self.gcn.eval()
        for data in valid_loader:
            outputs = self.gcn(data.x, data.edge_index)
            labels = data.y.unsqueeze(1)
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
        self.gcn.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for data in test_loader:
                outputs = self.gcn(data.x, data.edge_index)
                predicted = (outputs > 0.5).int()
                true_labels.extend(data.y.tolist())
                predictions.extend(predicted.cpu().numpy().flatten())

        return np.asarray(true_labels, dtype=np.int32), np.asarray(
            predictions, dtype=np.int32
        )

    def classify(self, feature_data, adj_matrices, labels, verbose=True):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.gcn.parameters(), lr=self.lr, weight_decay=self.wd)

        dataset = GraphDataset(feature_data, adj_matrices, labels)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_size = int(0.75 * train_size)
        valid_size = len(dataset) - train_size - test_size
        train_dataset, valid_dataset = random_split(
            train_dataset, [train_size, valid_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        self.train(criterion, optimizer, train_loader, val_loader, verbose=verbose)

        y_lab, preds = self.predict(test_loader)
        return y_lab, preds

    def hyperparameter_tuning(
        self, feature_data, adj_matrices, labels, multilayer=False, max_evals=25
    ):
        feature_data = np.array(feature_data)
        adj_matrices = np.array(adj_matrices)
        labels = np.array(labels)

        def objective(params):
            skf = StratifiedKFold(n_splits=5)
            val_accuracies = []

            for train_idx, val_idx in skf.split(feature_data, labels):
                self.gcn = GCN()

                # params["batch_size"] = int(params["batch_size"])
                self.lr = params["lr"]
                self.wd = params["weight_decay"]
                self.gamma = params["gamma"]

                X_train, adj_train, y_train = (
                    feature_data[train_idx],
                    adj_matrices[train_idx],
                    labels[train_idx],
                )
                X_val, adj_val, y_val = (
                    feature_data[val_idx],
                    adj_matrices[val_idx],
                    labels[val_idx],
                )

                train_dataset = GraphDataset(X_train, adj_train, y_train)
                val_dataset = GraphDataset(X_val, adj_val, y_val)
                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

                criterion = nn.BCELoss()
                optimizer = optim.Adam(
                    self.gcn.parameters(), lr=self.lr, weight_decay=self.wd
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

        # best["batch_size"] = int(best["batch_size"])

        self.lr = best["lr"]
        self.wd = best["weight_decay"]
        self.gamma = best["gamma"]

        optimal_hyperparameters = {
            "Learning Rate": [best["lr"]],
            "Weight Decay": [best["weight_decay"]],
            "Gamma": [best["gamma"]],
        }

        print(tabulate(optimal_hyperparameters, headers="keys", tablefmt="fancy_grid"))

    def cross_validate(self, feature_data, adj_matrices, labels, cv, verbose=False):
        kfold = StratifiedKFold(n_splits=cv, shuffle=True)
        accuracies = []

        tprs = []
        mean_fpr = np.linspace(0, 1, 1000)

        histories = []

        targets = []
        predictions = []

        feature_data = np.array(feature_data)
        adj_matrices = np.array(adj_matrices)
        labels = np.array(labels)

        for train_ids, val_ids in kfold.split(feature_data, labels):
            self.gcn = GCN()

            X_train, adj_train, y_train = (
                feature_data[train_ids],
                adj_matrices[train_ids],
                labels[train_ids],
            )
            X_val, adj_val, y_val = (
                feature_data[val_ids],
                adj_matrices[val_ids],
                labels[val_ids],
            )

            # Create GraphDataset for training and validation
            train_dataset = GraphDataset(X_train, adj_train, y_train)
            val_dataset = GraphDataset(X_val, adj_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(
                self.gcn.parameters(), lr=self.lr, weight_decay=self.wd
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


class StackingEnsemble(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_nodes, dropout_rate=0.5
    ):
        super(StackingEnsemble, self).__init__()
        self.gcns = nn.ModuleList(
            [
                GCN(in_channels, hidden_channels, out_channels, num_nodes, dropout_rate)
                for _ in range(5)
            ]
        )
        self.random_forest = RandomForestClassifier()
        self.bayes_optimal = GaussianNB()
        self.meta_model = LinearRegression()

    def forward(self, inputs, target_labels=None):
        outputs = [gcn(A, X) for gcn, (A, X) in zip(self.gcns, inputs)]
        outputs = torch.cat(outputs, dim=1).detach().numpy()

        # Training and predictions for base models
        if self.training:
            self.random_forest.fit(outputs, target_labels)
            self.bayes_optimal.fit(outputs, target_labels)

        rf_predictions = self.random_forest.predict(outputs)
        mv_predictions = stats.mode(outputs, axis=1)[0]
        bo_predictions = self.bayes_optimal.predict(outputs)

        # Combining predictions for meta-model
        combined_predictions = np.vstack(
            (rf_predictions, mv_predictions, bo_predictions)
        ).T
        if self.training:
            self.meta_model.fit(combined_predictions, target_labels)

        final_output = self.meta_model.predict(combined_predictions)

        return torch.tensor(final_output, dtype=torch.float32)


# # Example usage Simple GCN:
# # Define the node features matrix and edge index as PyTorch tensors
# num_nodes = 3
# num_features_per_node = 2
# X = torch.rand((num_nodes, num_features_per_node))  # Node feature matrix (NxD)
# edge_index = torch.tensor(
#     [[0, 1, 2, 0, 2], [1, 0, 0, 2, 1]], dtype=torch.long
# )  # Edge index

# # Initialize the GCN model
# gcn_model = GCN(in_channels=num_features_per_node, hidden_channels=2, out_channels=1)

# # Perform a forward pass through the GCN model
# output = gcn_model(X, edge_index)
# print(output)


# # Example usage Ensemble GCN:
# num_nodes = 3
# num_features_per_node = 2
# inputs = [
#     (torch.rand((num_nodes, num_nodes)), torch.rand((num_nodes, num_features_per_node)))
#     for _ in range(5)
# ]

# # Initialize the Ensemble GCN model
# ensemble_gcn_model = StackingEnsemble(
#     in_channels=num_features_per_node,
#     hidden_channels=2,
#     out_channels=1,
#     num_nodes=num_nodes,
# )

# # Perform a forward pass through the Ensemble GCN model
# output = ensemble_gcn_model(inputs)
# print(output)
