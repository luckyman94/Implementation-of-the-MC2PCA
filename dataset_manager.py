import os
import numpy as np
from sktime.datasets import load_japanese_vowels, load_basic_motions
from config import DATA


class DatasetManager:

    def __init__(self, name):
        self.name = name.lower()
        self.X = None
        self.y = None
        self._load()

    def _load(self):
        if self.name == "japanese_vowels":
            self._load_japanese_vowels()

        elif self.name == "spoken_arabic_digits":
            self._load_spoken_arabic_digits()
        
        elif self.name == "basicmotions":
            self._load_basic_motions()

        else:
            raise ValueError(f"Unknown dataset: {self.name}")

    def _load_basic_motions(self):
        X, y = load_basic_motions(return_X_y=True)
        self.X = self._sktime_to_matrices(X)
        self.y = y

    def _load_spoken_arabic_digits(self):
        root_dir = os.path.join(DATA, "spoken-arabic-digit")

        def read_file(path):
            series = []
            current = []

            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        if len(current) > 0:
                            series.append(np.array(current))
                            current = []
                    else:
                        current.append([float(v) for v in line.split()])

                if len(current) > 0:
                    series.append(np.array(current))

            return series

        X_train = read_file(os.path.join(root_dir, "Train_Arabic_Digit.txt"))
        X_test  = read_file(os.path.join(root_dir, "Test_Arabic_Digit.txt"))

        y_train = [d for d in range(10) for _ in range(len(X_train) // 10)]
        y_test  = [d for d in range(10) for _ in range(len(X_test)  // 10)]

        self.X = X_train + X_test
        self.y = np.array(y_train + y_test)

    def _load_japanese_vowels(self):
        X, y = load_japanese_vowels(return_X_y=True)
        self.X = self._sktime_to_matrices(X)
        self.y = y

    def _sktime_to_matrices(self, X):
        series_list = []
        for i in range(len(X)):
            cols = [X.iloc[i, j] for j in range(X.shape[1])]
            Xi = np.column_stack(cols)
            series_list.append(Xi)
        return series_list
