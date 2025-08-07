import json
import numpy as np


class OneHotEncoder:
    def __init__(self, categories=None, handle_unknown="ignore"):
        self.categories = categories
        if self.categories is None:
            self.class_to_index = {}
        else:
            self.class_to_index = {cls: idx for idx, cls in enumerate(self.categories)}
        self.handle_unknown = handle_unknown

    def fit(self, list_to_fit):
        if self.categories is not None:
            raise KeyError("categories already exist")
        self.categories = sorted(set(list_to_fit))
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.categories)}

    def transform(self, list_to_transf):
        indices = [self.class_to_index.get(label, -1) for label in list_to_transf]
        return self._to_one_hot(indices)

    def _to_one_hot(self, indices):
        one_hot = np.zeros((len(indices), len(self.categories)), dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx != -1:
                one_hot[i, idx] = 1.0
        return one_hot

    def save_json(self, saving_path):
        with open(saving_path, "w") as f:
            json.dump(self.categories, f)
