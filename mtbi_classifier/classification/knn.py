import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    auc,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils.plot_fig import (
    plot_roc_curve,
    plot_avg_roc_curve,
)


class KNN:
    def __init__(self, k=5):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def train(self, X_train, y_train):
        self.knn.fit(X_train, y_train)

    def predict(self, X_test):
        return self.knn.predict(X_test)

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

    def find_best_k(self, X, y, k_range, cv=5):
        """
        Find the best value of k using cross-validation.

        Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Label vector.
        k_range (range): Range of k values to test.
        cv (int): Number of folds in the cross-validation.
        """
        k_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(
                knn, X.to_numpy(), y.to_numpy(), cv=cv, scoring="accuracy"
            )
            k_scores.append(scores.mean())

        from matplotlib import pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=k_range, y=k_scores, marker="o")
        plt.xlabel("Value of K for KNN")
        plt.ylabel("Cross-Validated Accuracy")
        plt.title("Cross-Validation Accuracy as a function of K")

        optimal_k = k_range[k_scores.index(max(k_scores))]
        self.k = optimal_k
        self.knn = KNeighborsClassifier(n_neighbors=optimal_k)
        print(f"Optimal value of k: {optimal_k}")

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
            pred = self.predict(X_test.to_numpy())

            # Compute accuracy score
            scores.append(accuracy_score(y_test, pred))

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        # Print average accuracy
        scores = np.array(scores)
        print(
            f"Average Cross-Validation Score: {scores.mean()*100.0:.3f}% ± {scores.std()*100:.3f}"
        )

        # Plot ROC curve
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        plot_avg_roc_curve(mean_fpr, mean_tpr, std_tpr, auc(mean_fpr, mean_tpr))

    def permutation_test(self, X_test, y_test):
        """
        Perform a permutation test to identify important features.

        Parameters:
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test label vector.
        features (list): List of feature names to test.
        """
        features = X_test.columns

        # Calculate baseline metrics
        y_pred = self.predict(X_test)
        y_test = y_test.to_numpy()
        baseline_accuracy = accuracy_score(y_test, y_pred)

        important_features = {}

        for feature in features:
            X_permuted = X_test.copy()
            X_permuted[feature] = np.random.permutation(X_test[feature].to_numpy())

            y_pred_permuted = self.predict(X_permuted)
            permuted_accuracy = accuracy_score(y_test, y_pred_permuted)

            if permuted_accuracy < baseline_accuracy:
                important_features[feature] = {"Acc": permuted_accuracy}

        # Print important features
        print("Important features:")
        for feature, metrics in important_features.items():
            print(f"{feature}: {metrics}")


# Usage example:
# knn = KNN()
# knn.find_best_k(X, y, range(1, 31))
# knn.train(X_train, y_train)
# knn.permutation_test(X_test, y_test)
