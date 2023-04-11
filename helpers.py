# Initialize training data partitions for the dataloader
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
import os
import zipfile
from pathlib import Path

def load_data(filepath: str = 'data'):
    train = pd.read_csv('data/fashion-mnist_train.csv')
    test = pd.read_csv('data/fashion-mnist_test.csv')
    curated_train = train[(train.iloc[:, 0] == 0) | (train.iloc[:, 0] == 1)]
    curated_test = test[(test.iloc[:, 0] == 0) | (test.iloc[:, 0] == 1)]
    curated_train.to_csv('data/fashion-mnist_train_curated.csv', index=False)
    curated_test.to_csv('data/fashion-mnist_test_curated.csv', index=False)


def randomize_experiment(path='setup/minus_plus_split.xlsx'):
    df = pd.read_excel(path, header=0)
    df = df.sample(frac=1)
    df.columns = ["id", "learning_rate", "batch_size", "layer_size", "layer_number", "response"]
    print("Randomized DataFrame:\n", df)
    df.to_csv('setup/randomized_experiment.csv')

def zip_files():
    zip_path = "plots/myplot.zip"
    folder_path = 'plots'


    # Create a ZipFile object with write permission
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        # Iterate over all files in the folder
        for foldername, subfolders, filenames in os.walk(folder_path):
            # Add each file to the zip file
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))


if __name__ == "__main__":
    # zip_files()
    randomize_experiment()
    # load_data()
    # loaded = pd.read_csv('data/fashion-mnist_train_curated.csv')
    # y = loaded.iloc[:, 0].to_numpy()
    # X = loaded.drop(loaded.iloc[:, 0],).to_numpy()
    # print(y[:5])
    # print(X[:5])
