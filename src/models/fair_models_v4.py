from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
import pandas as pd


class FairClassifierV4(BaseEstimator, ClassifierMixin):
    """
    A fairness-aware scikit-learn compatible classifier that trains separate models for each sensitive value.
    Each model will then give prediction probabilities for each class of the instance. The probabilities for each class
    are then depending on the sensitive group which the instance belongs multiplied by accuracy of each model for the
    given sensitive group and summed together. The class with the highest value is then chosen as the predicted class.

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
    scores : dict
        A nested dictionary storing the accuracy scores of each model on all sensitive attribute groups.
    clf_type : class
        The class of the base estimator.
    clf_kwargs : dict
        Additional keyword arguments passed to the `clf_type` during model initialization.
    _classes : ndarray
        The unique class labels from the target variable `y`
    """

    def __init__(self, clf_type, **kwargs):
        self.models = {}
        self.scores = {}
        self.clf_type = clf_type
        self.clf_kwargs = kwargs
        self._classes = None

    def fit(self, X_A, y):
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
            self.scores[f'model_{value}'] = {}

            # Loop through all sensitive groups
            for _value in sensitive_values:
                # Filtering the data for the current sensitive group
                _X_group = X[A == _value]
                _y_group = y[A == _value]
                # Store the trained model scores for the current sensitive group
                self.scores[f'model_{value}'][f'{_value}'] = model.score(_X_group, _y_group)

        return self

    def predict(self, X_A):
        """
        Each trained model gives class probabilities for the given data (`X`). The probabilities for each class
        are then depending on the sensitive group which the instance belongs multiplied by accuracy of each model for the
        given sensitive group and summed together. The class with the highest value is then chosen as the predicted class.

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

        sensitive_values = np.unique(A)  # Unique categories of the sensitive variable
        for value in sensitive_values:
            if f'{value}' not in self.models:
                raise ValueError(f"No model found for sensitive value '{value}'.")

        # Initialization of array of summed prediction probabilities of each class for each instance
        pred_probas = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)

        # Loop through models for each sensitive group
        for sensitive_value in self.models:
            # The model trained on the current sensitive group gives prediction probabilities for each class for each instance
            preds_temp = self.models[f'{sensitive_value}'].predict_proba(X)
            # Initialize the weight vector of current model accuracy scores, given the sensitive value of each instance
            weight_vector = np.array([self.scores[f'model_{sensitive_value}'][f'{a_value}'] for a_value in A])
            # Prediction probabilities are then multiplied by weight vector of model accuracies and summed to the final prediction array
            pred_probas += preds_temp * weight_vector[:, np.newaxis]

        # Returning the class labels with the highest value for each instance
        return self._classes[np.argmax(pred_probas, axis=1)]

    @property
    def classes_(self):
        if self._classes is None:
            raise ValueError("No classes available. The model has not been trained yet.")
        return self._classes
