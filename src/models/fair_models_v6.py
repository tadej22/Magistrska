from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class FairClassifierV6(BaseEstimator, ClassifierMixin):
    """
    A fairness-aware scikit-learn compatible classifier that trains separate models for each sensitive value.
    In addition, a selector model is trained, which chooses which model in ensemble will be used for the given instance.

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
    model_selector : object
        Model which chooses which model in ensemble will be used for the given instance.
    clf_type : class
        The class of the base estimator.
    clf_kwargs : dict
        Additional keyword arguments passed to the `clf_type` during model initialization.
    _classes : ndarray
        The unique class labels from the target variable `y`
    """

    def __init__(self, clf_type, **kwargs):
        self.models = {}
        self.model_selector = None
        self.clf_type = clf_type
        self.clf_kwargs = kwargs
        self._classes = None

    def fit(self, X_Z, y):
        """
        Fits the model to the provided data, training one model per sensitive group.
        In addition, a selector model is trained, which chooses which model in ensemble will be used for the given instance.

        Parameters
        __________
        X_Z : tuple (`X`, `Z`)
            X : {array-like, sparse matrix} of shape (n_samples, n_features).
            Z : {array-like} of shape (n_samples,) (1-dimensional array-like object)
            Tuple that consists of features (`X`) and sensitive features (`Z`).

        y : {array-like} of shape (n_samples,)
            The target labels.

        Returns
        -------
        self : object
            Returns the fitted instance of custom classifier itself.
        """

        # Check if X_Z tuple has 2 elements (X and Z)
        if not (isinstance(X_Z, tuple) and len(X_Z) == 2):
            raise ValueError("X_Z must be a tuple with exactly two elements: (X, Z).")
        X, Z = X_Z
        # Check if Z is a 1D array
        if not (isinstance(Z, (list, tuple, np.ndarray, pd.Series)) and np.ndim(Z) == 1):
            raise ValueError("The sensitive attribute Z must be a 1-dimensional array-like object.")

        sensitive_values = np.unique(Z)  # Unique categories of the sensitive variable
        self._classes = np.unique(y)  # Unique target classes in training data

        for value in sensitive_values:
            # Filtering the data for the current sensitive group
            X_group = X[Z == value]
            y_group = y[Z == value]

            # Training a model for the current sensitive group
            model = self.clf_type(**self.clf_kwargs)
            model.fit(X_group, y_group)
            self.models[f'{value}'] = model  # Store the trained model for the current sensitive group

        # Training the selector model, which will choose which model in ensemble will give prediction for given instance
        model_selector = self.clf_type(**self.clf_kwargs)
        model_selector.fit(X, Z)
        self.model_selector = model_selector

        return self

    def predict(self, X):
        """
        Predicts class labels for the given data, where each model gives prediction for data (`X`) of the sensitive group
        that the model has been trained on.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features).
            Instances to predict.

        Returns
        -------
        ndarray
            Predicted class labels for each instance.
        """

        # The selector model selects which model in ensemble will give prediction for each instance
        selector_preds = self.model_selector.predict(X)

        predictions = np.empty(X.shape[0], dtype=np.int64)

        # Loop through all trained models in ensemble for each sensitive group
        for sensitive_value in self.models:
            mask = (selector_preds == sensitive_value)  # Mask to be applied on X, for the current sensitive variable (where Z == sensitive group)
            if np.any(mask):
                predictions[mask] = self.models[sensitive_value].predict(X[mask])  # Predictions on data of the current sensitive group

        # Returning predictions and array of sensitive values of models for each prediction
        return predictions, selector_preds

    @property
    def classes_(self):
        if self._classes is None:
            raise ValueError("No classes available. The model has not been trained yet.")
        return self._classes
