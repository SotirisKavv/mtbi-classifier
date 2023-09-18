import pandas as pd


def import_panda_csv(path):
    mylist = []

    for chunk in pd.read_csv(path, sep=";", lineterminator="\n", chunksize=5000):
        mylist.append(chunk)

    big_data = pd.concat(mylist, axis=0)
    del mylist

    return big_data
