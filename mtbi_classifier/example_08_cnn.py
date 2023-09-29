from utils.import_data import import_all_data
from classification.cnn import classifyCNN, hyperparameter_tuning, cross_validation

#   Example 02: Functional Connectivity Maps

import matplotlib.pyplot as plt
import seaborn as sns

#  import Utils
from utils.import_data import import_all_data


sns.set_theme(style="whitegrid")


graphs, labels = import_all_data("graphs/full_multilayer/IPLV")
graphs = [graph.to_numpy() for graph in graphs]

lr_search = [0.0015, 0.0017, 0.0018, 0.0019]
bs_search = [3, 4, 5, 6]
wd_search = [0.0, 0.00001, 0.0001]


# lr, bsize, w_decay = hyperparameter_tuning(
#     graphs, labels, lr_search, bs_search, wd_search, multilayer=True
# )
lr, bsize, w_decay = 0.0019, 5, 0.0
# y_test, predictions = classifyCNN(graphs, labels, lr, bsize, w_decay, plot=True)
# accuracy = (y_test == predictions).mean()
# print(f"Accuracy: {accuracy:.4f}")
cross_validation(graphs, labels, 5, lr, bsize, w_decay, multilayer=True)

plt.show()
