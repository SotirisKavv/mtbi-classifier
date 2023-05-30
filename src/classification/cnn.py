# classification using Convolutional Neural Network
import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split


class Net(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.classifier = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = torch.mean(x, dim=0)  # Mean pooling
        x = self.classifier(x)

        return F.log_softmax(x, dim=-1)


# Assuming data_list is a list of PyTorch Geometric data objects, and labels is a list of your labels


def classifyCNN(data, labels):
    # Split data into train and test
    data_train, data_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    # Convert lists to DataLoader
    loader_train = DataLoader(data_train, batch_size=32, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=32)

    # Initialize the network and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_node_features=data[0].num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the network
    model.train()
    for _ in range(100):  # 100 epochs
        for batch in loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()

    # Test the network
    model.eval()
    predictions, y_true = [], []
    for batch in loader_test:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).max(dim=1)[1]
        predictions.extend(pred.cpu().numpy())
        y_true.extend(batch.y.cpu().numpy())

    return y_true, predictions
