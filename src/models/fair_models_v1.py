from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class FairClassifierV1(BaseEstimator, ClassifierMixin):
    """
    A fairness-aware scikit-learn compatible classifier that trains separate models for each sensitive value.
    Given the sensitive value of the instance, the corresponding model fitted on data of the sensitive group
    will give the prediction.

    Parameters
    __________
    clf_type : class
        The class of the base estimator (e.g., sklearn classifiers like `RandomForestClassifier`).
    **kwargs : dict
        Additional keyword arguments passed to the `clf_type` during model initialization.

    Attributes
    __________
    models : dict
        A dictionary where the keys are sensitive values and the values are the trained models for those sensitive groups.
    clf_type : class
        The class of the base estimator.
    clf_kwargs : dict
        Additional keyword arguments passed to the `clf_type` during model initialization.
    _classes : ndarray
        The unique class labels from the target variable `y`
    """

    def __init__(self, clf_type, **kwargs):
        self.models = {}
        self.clf_type = clf_type
        self.clf_kwargs = kwargs
        self._classes = None

    def fit(self, X_A, y):
        """
        Fits the model to the provided data, training one model per sensitive group.

        Parameters
        __________
        X_A : tuple (`X`, `A`)
            X : {array-like, sparse matrix} of shape (n_samples, n_features).
            A : {array-like} of shape (n_samples,) (1-dimensional array-like object)
            Tuple that consists of features (`X`) and sensitive features (`A`).

        y : {array-like} of shape (n_samples,)
            The target labels.

        Returns
        -------
        self : object
            Returns the fitted instance of custom classifier itself.
        """

        # Check if X_A tuple has 2 elements (X and A)
        if not (isinstance(X_A, tuple) and len(X_A) == 2):
            raise ValueError("X_A must be a tuple with exactly two elements: (X, A).")
        X, A = X_A
        # Check if A is a 1D array
        if not (isinstance(A, (list, tuple, np.ndarray, pd.Series)) and np.ndim(A) == 1):
            raise ValueError("The sensitive attribute A must be a 1-dimensional array-like object.")

        sensitive_values = np.unique(A)  # Unique categories of the sensitive variable
        self._classes = np.unique(y)  # Unique target classes in training data

        for value in sensitive_values:
            # Filtering the data for the current sensitive group
            X_group = X[A == value]
            y_group = y[A == value]

            # Training a model for the current sensitive group
            model = self.clf_type(**self.clf_kwargs)
            model.fit(X_group, y_group)
            self.models[f'{value}'] = model  # Store the trained model for the current sensitive group

        return self

    def predict(self, X_A):
        """
        Predicts class labels for the given data, where each model gives prediction for data (`X`) of the sensitive group
        that the model has been trained on.

        Parameters
        ----------
        X_A : tuple (`X`, `A`)
            X : {array-like, sparse matrix} of shape (n_samples, n_features).
            A : {array-like} of shape (n_samples,) (1-dimensional array-like object)
            Tuple that consists of features (`X`) and sensitive features (`A`).

        Returns
        -------
        ndarray
            Predicted class labels for each instance.
        """

        # Check if X_A tuple has 2 elements (X and A)
        if not (isinstance(X_A, tuple) and len(X_A) == 2):
            raise ValueError("X_A must be a tuple with exactly two elements: (X, A).")
        X, A = X_A
        # Check if A is a 1D array
        if not (isinstance(A, (list, tuple, np.ndarray, pd.Series)) and np.ndim(A) == 1):
            raise ValueError("The sensitive attribute A must be a 1-dimensional array-like object.")

        predictions = np.empty(X.shape[0], dtype=np.int64)
        sensitive_values = np.unique(A)  # Unique categories of the sensitive variable

        for value in sensitive_values:
            if f'{value}' not in self.models:
                raise ValueError(f"No model found for sensitive value '{value}'.")
            mask = (A == value)  # Mask to be applied on X, for the current sensitive variable (where A == sensitive group)
            predictions[mask] = self.models[value].predict(X[mask])  # Predictions on data of the current sensitive group

        return predictions

    @property
    def classes_(self):
        if self._classes is None:
            raise ValueError("No classes available. The model has not been trained yet.")
        return self._classes
