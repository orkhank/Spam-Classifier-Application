from abc import ABC, abstractmethod
from typing import Union
from numpy import ndarray

from sklearn.pipeline import Pipeline


class Classifier(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_parameters(self) -> None:
        pass

    def fit(self, X, y, **fit_params):
        # st.write(X, y, self.clf)
        assert hasattr(self, "clf"), f"{self} does not have the `clf` attribute."
        assert self.clf is not None, "Pipeline has not been initialized."
        self.clf.fit(X, y, **fit_params)
        return self.clf

    def predict(self, X):
        assert hasattr(self, "clf"), f"{self} does not have the `clf` attribute."
        assert self.clf is not None and isinstance(
            self.clf, Pipeline
        ), "Pipeline has not been initialized."
        assert self.clf.__sklearn_is_fitted__(), "The Pipeline is not fitted."
        return self.clf.predict(X)

    # TODO: add method for hyperparameter optimization
