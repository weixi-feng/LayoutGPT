# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import csv
import numpy as np

class SplitsBuilder(object):
    def __init__(self, train_test_splits_file):
        self._train_test_splits_file = train_test_splits_file
        self._splits = {}

    def train_split(self):
        return self._splits["train"]

    def test_split(self):
        return self._splits["test"]

    def val_split(self):
        return self._splits["val"]

    def _parse_train_test_splits_file(self):
        with open(self._train_test_splits_file, "r") as f:
            data = [row for row in csv.reader(f)]
        return np.array(data)

    def get_splits(self, keep_splits=["train, val"]):
        if not isinstance(keep_splits , list):
            keep_splits = [keep_splits]
        # Return only the split
        s = []
        for ks in keep_splits:
            s.extend(self._parse_split_file()[ks])
        return s


class CSVSplitsBuilder(SplitsBuilder):
    def _parse_split_file(self):
        if not self._splits:
            data = self._parse_train_test_splits_file()
            for s in ["train", "test", "val"]:
                self._splits[s] = [r[0] for r in data if r[1] == s]
        return self._splits
