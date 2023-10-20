import numpy as np
from scipy.stats import ttest_1samp
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    auc,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from utils.plot_fig import (
    plot_roc_curve,
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
            accs = []
            self.knn = KNeighborsClassifier(n_neighbors=k)
            for _ in range(300):
                acc, _ = self.cross_validate(X, y, cv=cv)
                accs.append(acc[0])
            k_scores.append(np.array(accs).mean())

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
        kf = StratifiedKFold(n_splits=cv, shuffle=True)
        scores = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 1000)

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train = np.ascontiguousarray(X_train)
            X_test = np.ascontiguousarray(X_test)

            self.train(X_train, y_train)
            pred = self.predict(X_test)

            # Compute accuracy score
            scores.append(accuracy_score(y_test, pred))

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        # Print average accuracy
        scores = np.array(scores)

        acc_params = scores.mean(), scores.std()

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        auc_roc = auc(mean_fpr, mean_tpr)

        roc_params = mean_fpr, mean_tpr, std_tpr, auc_roc

        return acc_params, roc_params

    def permutation_t_test(self, X_test, y_test, n_permutations=1000, alpha=0.05):
        """
        Perform a permutation t-test to identify important features.

        Parameters:
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): Test label vector.
        n_permutations (int): Number of permutations for each feature.
        """
        features = X_test.columns

        X_test = np.ascontiguousarray(X_test)
        y_pred = self.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred)

        t_statistics = {}
        p_values = {}

        for feature in features:
            # Permute the feature values and compute the t-statistic for each permutation
            permuted_accuracies = []
            for _ in range(n_permutations):
                X_permuted = X_test.copy()
                X_permuted[:, features.get_loc(feature)] = np.random.permutation(
                    X_test[:, features.get_loc(feature)]
                )
                y_pred_permuted = self.predict(X_permuted)
                permuted_accuracy = accuracy_score(y_test, y_pred_permuted)
                permuted_accuracies.append(permuted_accuracy)

            # Compute the p-value
            t_stat, p_value = ttest_1samp(permuted_accuracies, baseline_accuracy)
            t_statistics[feature] = t_stat
            p_values[feature] = p_value

        important_features = [
            feature for feature, p_value in p_values.items() if p_value < alpha
        ]

        t_statistics = {
            feature: t_stat
            for feature, t_stat in t_statistics.items()
            if feature in important_features
        }
        p_values = {
            feature: p_value
            for feature, p_value in p_values.items()
            if feature in important_features
        }
        # Sort the features by their t-statistic
        sorted_p = sorted(p_values.items(), key=lambda x: x[1], reverse=True)

        return t_statistics, sorted_p


# Usage example:
# knn = KNN()
# knn.find_best_k(X, y, range(1, 31))
# knn.train(X_train, y_train)
# knn.permutation_test(X_test, y_test)
