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
