import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_max_pool, BatchNorm
from torch_geometric.data import Data, DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


class GraphDataset(Dataset):
    def __init__(self, X, edge_index, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Node features
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)  # Graph structure
        self.y = torch.tensor(y, dtype=torch.float32)  # Labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.edge_index[idx], self.y[idx]


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
        self.conv1 = GCNConv(100, 1024)
        self.conv2 = GCNConv(1024, 1024)
        self.batch_norm1 = BatchNorm(100)
        self.batch_norm2 = BatchNorm(1024)
        self.batch_norm3 = BatchNorm(1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 100)
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


# Example usage Simple GCN:
# Define the node features matrix and edge index as PyTorch tensors
num_nodes = 3
num_features_per_node = 2
X = torch.rand((num_nodes, num_features_per_node))  # Node feature matrix (NxD)
edge_index = torch.tensor(
    [[0, 1, 2, 0, 2], [1, 0, 0, 2, 1]], dtype=torch.long
)  # Edge index

# Initialize the GCN model
gcn_model = GCN(in_channels=num_features_per_node, hidden_channels=2, out_channels=1)

# Perform a forward pass through the GCN model
output = gcn_model(X, edge_index)
print(output)


# Example usage Ensemble GCN:
num_nodes = 3
num_features_per_node = 2
inputs = [
    (torch.rand((num_nodes, num_nodes)), torch.rand((num_nodes, num_features_per_node)))
    for _ in range(5)
]

# Initialize the Ensemble GCN model
ensemble_gcn_model = StackingEnsemble(
    in_channels=num_features_per_node,
    hidden_channels=2,
    out_channels=1,
    num_nodes=num_nodes,
)

# Perform a forward pass through the Ensemble GCN model
output = ensemble_gcn_model(inputs)
print(output)
