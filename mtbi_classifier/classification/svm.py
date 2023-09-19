import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    auc,
    roc_curve,
)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from utils.plot_fig import (
    plot_roc_curve,
    plot_avg_roc_curve,
)


class SVM:
    def __init__(self):
        self.svm = SVC(kernel="linear")

    def train(self, X_train, y_train):
        self.svm.fit(X_train, y_train)

    def predict(self, X_test):
        return self.svm.predict(X_test)

    def score(self, y_test, pred):
        accuracy = accuracy_score(y_test, pred)
        print(f"Accuracy: {accuracy}")

        # Calculate confusion_matrix
        confusion = confusion_matrix(y_test, pred)
        print(f"Confusion Matrix:\n{confusion}")

        # Calculate classification_report
        classification = classification_report(y_test, pred, zero_division=0)
        print(f"classification Report:\n{classification}")

        # Calculate AUC-ROC
        auc_roc = roc_auc_score(y_test, pred)
        print(f"AUC-ROC: {auc_roc}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, pred)
        plot_roc_curve(fpr, tpr)

    def cross_validate(self, X, y, cv=5):
        """
        Perform k-fold cross-validation and print the average accuracy across all folds.

        Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Label vector.
        cv (int): Number of folds in the cross-validation.
        """
        kf = StratifiedKFold(n_splits=cv)
        scores = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.train(X_train, y_train)
            pred = self.predict(X_test)

            # Compute accuracy score
            scores.append(accuracy_score(y_test, pred))

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        # Print average accuracy
        scores = np.array(scores)
        print(
            f"Average Cross-Validation Score: {scores.mean():.3f} ± {scores.std():.3f}"
        )

        # Plot ROC curve
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        plot_avg_roc_curve(mean_fpr, mean_tpr, std_tpr, auc(mean_fpr, mean_tpr))
