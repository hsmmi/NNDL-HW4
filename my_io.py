from glob import glob
import os
import pickle
import pandas as pd
from scipy.io import wavfile
import numpy as np


dataset_path = "./Dataset/"


class my_io:
    def __init__(self, dataset_folder_path=None):
        if dataset_folder_path is not None:
            self.dataset_folder_path = dataset_folder_path
        self.data = []

    def read_csv_fide(self, csv_file_path, col_label):
        return pd.read_csv(csv_file_path, header=None).to_numpy()[:, col_label]

    def read_csv_folder(self, csv_folder_path):
        # list all csv files in the folder
        csv_files = glob(csv_folder_path + "*.csv")

        # Sort by name
        csv_files.sort()

        # read all csv files
        data = pd.read_csv(csv_files[0], header=None).to_numpy()
        for i in range(1, len(csv_files)):
            csv_file = pd.read_csv(csv_files[i], header=None).to_numpy()
            data = np.append(data, csv_file, axis=0)

        return np.array(data)

    def read_wav_folder(self, wav_folder_path):
        # list all wav files in the folder
        wav_files = glob(wav_folder_path + "*.wav")

        # sort by name
        wav_files.sort()

        # read all wav files
        shortest_len = np.inf
        data = []
        for i in range(0, len(wav_files)):
            wav_file = wavfile.read(wav_files[i])[1]
            if len(wav_file) < shortest_len:
                shortest_len = len(wav_file)
            # Add a sample to data
            data.append([wav_file][0])

        # Reshape data
        np_data = np.empty((len(data), shortest_len))
        for i in range(0, len(data)):
            np_data[i] = data[i][:shortest_len]

        return np_data

    def Normalize_data(self, data, min_value, max_value):
        """
        Normalize data to [min_value, max_value]
        """

        new_data = np.empty_like(data)

        max_data_value = np.max(data)
        min_data_value = np.min(data)
        range_value = max_data_value - min_data_value
        range_normal = max_value - min_value
        for i in range(len(data)):
            normal_data = (
                data[i] - min_data_value
            ) / range_value * range_normal + min_value
            new_data[i] = normal_data

        return new_data

    def save_data(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data

    def read_file(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.readlines()
        else:
            return None
