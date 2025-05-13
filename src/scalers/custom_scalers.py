from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class BaseCustomScaler(BaseEstimator, TransformerMixin):
    """
    A custom scaler wrapper for scaling features while preserving associated metadata.

    This class wraps around a scikit-learn scaler (e.g., `MinMaxScaler`, `StandardScaler`)
    to allow transformations on features (`X_data`) while keeping additional metadata
    (e.g., sensitive attributes `Z_data`) unmodified and returned alongside the scaled features.
    Useful for including it as a step inside scikit-learn `Pipeline` together with custom fairness-aware
    estimators, where methods like `fit` and `transform` need to operate on both the feature data
    (`X_data`) and sensitive attributes (`Z_data`).

    Parameters
    ----------
    scaler_class : class
        The class of the scikit-learn scaler to be used (e.g., `MinMaxScaler`, `StandardScaler`).
    **kwargs : dict
        Additional keyword arguments passed to the `scaler_class` during initialization.

    Attributes
    ----------
    scaler_class : class
        The scaler class used for scaling the features.
    scaler_kwargs : dict
        The additional parameters passed to the scaler during initialization.
    scaler : object
        The instantiated scaler object after fitting.
    """

    def __init__(self, scaler_class, **kwargs):
        self.scaler_class = scaler_class
        self.scaler_kwargs = kwargs
        self.scaler = None

    def fit(self, X, y=None):
        """
        Fits the scaler to the feature data (`X_data`).

        Parameters
        ----------
        X : tuple (`X_data`, `Z_data`)
            X_data : {array-like, sparse matrix} of shape (n_samples, n_features).
            Tuple that consists of features (`X_data`) and sensitive features (`Z_data`).
            Present for compatibility with scikit-learn custom fairness-aware estimators.
            The scaler is fitted on the feature data (`X_data`).

        y : None
            Ignored, present for consistency.

        Returns
        -------
        self : object
            Fitted scaler.
        """

        X_data, Z_data = X
        self.scaler = self.scaler_class(**self.scaler_kwargs)
        self.scaler.fit(X_data, y)
        return self

    def transform(self, X):
        """
        Transforms the feature data (`X_data`) using the fitted scaler while preserving metadata (`Z_data`).

        Parameters
        ----------
        X : array-like or tuple (`X_data`, `Z_data`)
            - If array-like: Treated as X_data: {array-like, sparse matrix} of shape (n_samples, n_features).
            - If tuple: Tuple that consists of features (`X_data`) and sensitive features (`Z_data`).
            Present for compatibility with scikit-learn custom fairness-aware estimators.
            The data (`X_data`) used to scale along the features axis.

        Returns
        -------
        If input was tuple:
        (`X_data`, `Z_data`) : tuple
            Tuple of transformed array of features (`X_data`) and preserved sensitive features (`Z_data`).
        If input was array-like:
        `X_data` : array-like
            Transformed array of features.
        """

        if isinstance(X, tuple):
            X_data, Z_data = X
            X_scaled = self.scaler.transform(X_data)
            return X_scaled, Z_data
        else:
            return self.scaler.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits the scaler to the feature data (`X_data`) and then transforms the data (`X_data`).

        Parameters
        ----------
        X : tuple (`X_data`, `Z_data`)
            X_data : {array-like, sparse matrix} of shape (n_samples, n_features).
            Tuple that consists of features (`X_data`) and sensitive features (`Z_data`).
            Present for compatibility with scikit-learn custom fairness-aware estimators.

        y : None
            Ignored, present for consistency.

        **fit_params : dict
            Ignored, present for consistency.

        Returns
        -------
        (`X_data`, `Z_data`) : tuple
            Tuple of transformed array (`X_data`) and preserved sensitive features (`Z_data`).
        """

        return self.fit(X, y).transform(X)


class CustomStandardScaler(BaseCustomScaler):
    def __init__(self, **kwargs):
        super().__init__(scaler_class=StandardScaler, **kwargs)


class CustomMinMaxScaler(BaseCustomScaler):
    def __init__(self, **kwargs):
        super().__init__(scaler_class=MinMaxScaler, **kwargs)
