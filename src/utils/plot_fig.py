from math import sqrt, ceil, floor
import seaborn as sns
import matplotlib.pyplot as plt


def plot_fig(kind, figsz, data, title=None, xlabel=None, ylabel=None, suptitle=None):
    if kind == "heatmap":
        plt.figure(figsize=figsz)
        sns.heatmap(data, cmap="viridis")
        if title != None:
            plt.title(title)
        if xlabel != None:
            plt.xlabel(xlabel)
        if ylabel != None:
            plt.ylabel(ylabel)
    elif kind == "multiple_hm":
        plt.figure(figsize=figsz)
        if suptitle != None:
            plt.suptitle(suptitle)

        len_x = floor(sqrt(len(data)))
        len_y = ceil(sqrt(len(data)))

        for i in range(len_y):
            for j in range(len_x):
                if i * len_x + j >= len(data):
                    break
                plt.subplot(len_y, len_x, i * len_x + j + 1)
                sns.heatmap(data[i * len_x + j], cmap="viridis")
                plt.title(title[i * len_x + j])
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)

        plt.tight_layout()


def plot_accuracies(history):
    """Plot the history of accuracies"""
    accuracies = [x["val_acc"] for x in history]
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. No. of epochs")


def plot_losses(history):
    """Plot the losses in each epoch"""
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")
