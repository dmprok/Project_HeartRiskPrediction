import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            transformed_X = X.copy()
        elif isinstance(X, np.ndarray):
            transformed_X = pd.DataFrame(X)
        else:
            raise ValueError("Unsupported input type")

        for col in transformed_X.columns:
            mapper = self.mapping_dict[col]
            if mapper is not None:
                transformed_X[col] = transformed_X[col].apply(lambda x: mapper.get(x) if x in mapper else x)

        return transformed_X.values