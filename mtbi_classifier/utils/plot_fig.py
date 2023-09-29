import os
import numpy as np
import seaborn as sns
from scipy.fft import fft
import matplotlib.pyplot as plt
from matplotlib.path import Path
from math import sqrt, ceil, floor
from utils.constants import SAMPLE_RATE
from matplotlib.patches import PathPatch
from visbrain.objects import ConnectObj, SourceObj
from visbrain.gui import Brain

sns.set_theme(style="whitegrid")


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

    elif kind == "ps_single":
        plt.figure(figsize=figsz)
        fft_signal = fft(data)
        N = len(fft_signal)
        n = np.arange(N)
        freq = n * (SAMPLE_RATE / N)
        half_freq = freq[freq <= 80]
        plt.stem(half_freq, np.abs(fft_signal[freq <= 80]), markerfmt=" ", basefmt="-b")
        plt.title(title)  #  Original Signal Spectrum
        plt.xlabel(xlabel)  #    Frequency [Hz]
        plt.ylabel(ylabel)  #    Magnitude

    elif kind == "ps_multiple":
        plt.figure(figsize=figsz)
        if suptitle != None:
            plt.suptitle(suptitle)

        len_x = ceil(sqrt(len(data)))
        len_y = floor(sqrt(len(data)))

        for i in range(len_y):
            for j in range(len_x):
                if i * len_x + j >= len(data):
                    break
                plt.subplot(len_y, len_x, i * len_x + j + 1)
                fft_signal = fft(data[i * len_x + j])
                N = len(fft_signal)
                n = np.arange(N)
                freq = n * (SAMPLE_RATE / N)
                half_freq = freq[freq <= 80]

                plt.stem(
                    half_freq,
                    np.abs(fft_signal[freq <= 80]),
                    markerfmt=" ",
                    basefmt="-b",
                )
                plt.title(title[i * len_x + j])
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)

        plt.tight_layout()


def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")


def plot_avg_roc_curve(mean_fpr, mean_tpr, std_tpr, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_fpr, mean_tpr, color="b", label="Mean ROC curve (area = %0.2f)" % auc_score
    )
    plt.fill_between(
        mean_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        color="b",
        alpha=0.2,
        label="± std. dev.",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")


def plot_accuracies(history):
    """Plot the history of accuracies"""
    train_acc = [x.get("train_acc") for x in history]
    val_acc = [x["val_acc"] for x in history]
    plt.figure()
    plt.plot(train_acc, "-bx")
    plt.plot(val_acc, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["Training", "Validation"])
    plt.title("Accuracy vs. No. of epochs")


def plot_losses(history):
    """Plot the losses in each epoch"""
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    plt.figure()
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")


def plot_circular_chord(matrix, labels=None):
    """
    Plots a circular chord plot based on an adjacency matrix.

    Parameters:
    matrix (numpy.ndarray): The adjacency matrix representing the functional connectivity.
    labels (list): A list of labels for the nodes. Default is None.

    Returns:
    None
    """
    if labels is None:
        labels = [str(i).replace("\r", "") for i in range(len(matrix))]

    n = len(matrix)
    angle = 2 * np.pi / n
    angles = [i * angle for i in range(n)]

    plt.figure()
    _, ax = plt.subplots(subplot_kw=dict(polar=True))

    # Plot nodes
    ax.scatter(angles, [1] * n, c="grey", s=100)

    # Add labels
    for i, label in enumerate(labels):
        rotation = np.degrees(angles[i])
        alignment = "left"
        radial_distance = 1.1
        if angles[i] > np.pi / 2 and angles[i] < 3 * np.pi / 2:
            rotation += 180
            alignment = "right"
            radial_distance = 1.1
        ax.text(
            angles[i],
            radial_distance,
            label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=9,
        )

    # Get the colormap
    cmap = plt.cm.plasma

    # Get the maximum connection strength to normalize the colors
    max_strength = np.max(matrix)

    # Create a list of connections with their strengths
    connections = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0:
                connections.append((i, j, matrix[i][j]))

    # Sort the connections in ascending order of strength
    connections.sort(key=lambda x: x[2])

    # Plot chords
    for connection in connections:
        i, j, strength = connection
        if strength > 0.1:
            color = cmap(strength / max_strength)
            alpha = strength / max_strength

            # Create a Bezier curve
            verts = [
                (angles[i], 1),
                (angles[i], 1 - alpha / 2),
                (angles[j], 1 - alpha / 2),
                (angles[j], 1),
            ]

            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

            path = Path(verts, codes)
            patch = PathPatch(
                path, facecolor="none", lw=2, edgecolor=color, alpha=alpha
            )
            ax.add_patch(patch)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)

    # Add a colorbar
    norm = plt.Normalize(0, max_strength)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.12)


def show_brain_connectivity(adj_matrix, coords):
    s_obj = SourceObj("default", np.array(coords), color="red", radius_min=10)
    adj_matrix[np.tril_indices_from(adj_matrix)] = 0
    c_default = ConnectObj(
        "default", np.array(coords), adj_matrix, color_by="strength", cmap="plasma"
    )

    vb = Brain(source_obj=s_obj, connect_obj=c_default)

    # vb.connect_control(
    #     name="default",
    #     show=True,
    #     cmap="viridis",
    #     colorby="density",
    #     clim=(0.0, 10.0),
    #     vmin=0.0,
    #     vmax=10.0,
    #     dynamic=(0.1, 1.0),
    # )
    # save_as = os.path.join("./screenshots", "3_main_brain.png")
    # vb.screenshot(save_as, dpi=300, print_size=(10, 10), autocrop=True)

    # vb.cortical_projection(
    #     clim=(0, 50),
    #     cmap="Spectral_r",
    #     vmin=10.1,
    #     under="black",
    #     vmax=41.2,
    #     over="green",
    # )
    # vb.rotate(custom=(-90.0, 0.0))  # Rotate the brain

    vb.show()
