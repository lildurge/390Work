import numpy as np
import h5py
import os
import pandas as pd
import glob

filename = "./data.h5"
folders = ["James' Data", "Roy's Data", "Sophie's Data"]
people = ["James", "Roy", "Sophie"]

with h5py.File(filename, "w") as hdf:
    for i, person in enumerate(people):
        group = hdf.create_group(person)
        for j, file in enumerate(glob.glob(folders[i] + "/*.csv")):
            data = pd.read_csv(file)
            dataset_name = os.path.splitext(os.path.basename(file))[0]
            dataset = group.create_dataset(dataset_name, data=data.iloc[1:, 1:-1].values)

    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")

    dataset_group = hdf.create_group("dataset")
    train_group = dataset_group.create_group("train")
    test_group = dataset_group.create_group("test")

    test_dataset = test_group.create_dataset("test_dataset", data=test_df.values)
    train_dataset = train_group.create_dataset("train_dataset", data=train_df.values)
