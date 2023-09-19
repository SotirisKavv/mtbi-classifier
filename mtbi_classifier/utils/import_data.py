import pandas as pd
import os


def import_panda_csv(path):
    mylist = []

    for chunk in pd.read_csv(path, sep=";", lineterminator="\n", chunksize=5000):
        mylist.append(chunk)

    big_data = pd.concat(mylist, axis=0)
    del mylist

    return big_data


def import_all_data(path):
    dataset = []
    labels = []
    for root, _, file in os.walk(path):
        for f in file:
            print(f"Importing {f}...", end="\r")
            label = 1 if "TBI" in f else 0
            meg = import_panda_csv(os.path.join(root, f))
            dataset.append(meg)
            labels.append(label)

            print(end="\x1b[2K")

    print("Data loaded successfully!")
    return dataset, labels
