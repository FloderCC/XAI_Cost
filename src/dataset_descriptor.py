"""File description:

Script for calculating the dimensions and class balancing of datasets.
"""

import os

import pandas as pd

dataset_setup_list = [
    ["UNAC", []]
]
def run(dataset):
    global results_df

    dataset_name = dataset[0]

    print(f"Started analyzer with dataset {dataset_name}")

    # loading thr dataset
    dataset_folder = f"./datasets/{dataset_name}"

    df = pd.read_csv(f"{dataset_folder}/{os.listdir(dataset_folder)[0]}")

    print("----------------------------------------")
    print("\nDataset: ", dataset[0])

    print("Size:")
    print(df.shape)

    print("Info:")
    print(df.info())


for dataset_setup in dataset_setup_list:
    run(dataset_setup)