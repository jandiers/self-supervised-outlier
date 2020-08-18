import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_array, check_is_fitted


def _create_noise(n_samples, n_features):
    return np.random.rand(n_samples, n_features)


def _merge(X, noise):
    return np.vstack((X, noise))


def _get_labels(n_normal, data_size):
    n_noise = data_size - n_normal
    return np.hstack((np.zeros(n_normal), np.ones(n_noise)))


class IterativeNoiseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_sampling_iter: int = 20,
                 outlier_inference: bool = False,
                 verbose=False):

        if n_sampling_iter == 0:
            raise ValueError('n_sampling_iter must be at least 1.')

        self.n_sampling_iter = n_sampling_iter
        self.outlier_inference = outlier_inference
        self.verbose = verbose

        self.clf = RandomForestClassifier(min_samples_leaf=5)

    def fit(self, X, y=None):
        X = check_array(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        # Check if fit had been called
        check_is_fitted(self, 'is_fitted_')

        # Input validation
        X = check_array(X, accept_sparse=False)

        if self.outlier_inference:
            # add no noise when detect outliers
            return X

        n_samples, n_features = X.shape
        n_noise = n_samples
        full_noise = []

        # create noise
        clf = RandomForestClassifier(min_samples_leaf=5)
        for iteration in range(0, self.n_sampling_iter):
            noise = _create_noise(n_noise, n_features)
            X_i = _merge(X, noise)
            y_i = _get_labels(n_samples, X_i.shape[0])
            p = clf.fit(X_i, y_i).predict(X_i)
            noise = noise[p[n_samples:] == 1]
            full_noise.append(noise)

        noise = np.vstack(full_noise)

        X = np.vstack((X, noise))
        return X
