from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from classification.wlkernel import classifyWL
from classification.lapdec import classifyLD
from classification.cnn import classifyCNN


def classify(graphs, labels, method):
    if method == "WLKernel":
        y_test, predictions = classifyWL(graphs, labels)
    elif method == "LDecomp":
        y_test, predictions = classifyLD(graphs, labels)
    elif method == "CNN":
        y_test, predictions = classifyCNN(graphs, labels)

    print_results(y_test, predictions)


def print_results(test, predictions):
    # Calculate accuracy
    print("Results:")
    print("Y_test:", test)
    print("Predictions:", predictions)
    accuracy = accuracy_score(test, predictions)
    print(f"Accuracy: {accuracy}")

    # Calculate confusion_matrix
    confusion = confusion_matrix(test, predictions)
    print(f"Confusion Matrix:\n{confusion}")

    # Calculate classification_report
    classification = classification_report(test, predictions, zero_division=0)
    print(f"classification Report:\n{classification}")

    # Calculate AUC-ROC
    auc_roc = roc_auc_score(test, predictions)
    print(f"AUC-ROC: {auc_roc}")
