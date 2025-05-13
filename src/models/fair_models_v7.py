from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class FairClassifierV7(BaseEstimator, ClassifierMixin):
    """
    A fairness-aware scikit-learn compatible classifier that trains separate models for each sensitive value.
    After training each model, prediction probabilities of each model are collected. The model of class with the highest
    probability of any model is chosen as the new target variable. The final selector model is then trained on training
    data with new target variables.

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
        The final model which is trained on training data with new target variables.
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
        After training each model, prediction probabilities of each model are collected. The model of class with the
        highest probability of any model is chosen as the new target variable. The final model is then trained on
        training data with new target variables.

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

        # Initialization of 3d array, which will contain pred_probas of each model for each training instance
        pred_probas = np.empty((X.shape[0], len(self.models), len(self._classes)), dtype=np.float64)

        # Loop through all trained models in ensemble for each sensitive group
        for i, sensitive_value in enumerate(self.models):
            # The model trained on the current sensitive group gives prediction probabilities for each class for each instance
            preds_temp = self.models[f'{sensitive_value}'].predict_proba(X)
            pred_probas[:, i, :] = preds_temp

        # The model labels of models that have the highest value of predict_proba for each instance,
        # are selected as target variables for model selector
        w = np.argmax(np.max(pred_probas, axis=2), axis=1)
        sensitive_values = list(self.models.keys())
        model_labels = [sensitive_values[i] for i in w]

        # Training the final model on training data with new target variables
        model_selector = self.clf_type(**self.clf_kwargs)
        model_selector.fit(X, model_labels)
        self.model_selector = model_selector

        return self

    def predict(self, X):
        """
        The model selector, which is trained on labels of model based on prediction probabilities of models in ensemble
        on training data, chooses the most suitable model to return the final prediction for the given instance.

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
