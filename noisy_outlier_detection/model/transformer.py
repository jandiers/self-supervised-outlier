import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_random_state
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
                 verbose=False, n_jobs=None, random_state=None):

        if n_sampling_iter == 0:
            raise ValueError('n_sampling_iter must be at least 1.')

        self.n_sampling_iter = n_sampling_iter
        self.outlier_inference = outlier_inference
        self.verbose = verbose
        self.random_state = random_state
        self.random_state_ = None

        # hyperparameter search for self supervised task
        self.params = {'ccp_alpha': stats.uniform(loc=0, scale=0.1),
                       'min_samples_leaf': stats.randint(3, 10)}

        self.rndsearch_clf = RandomizedSearchCV(RandomForestClassifier(
            class_weight='balanced'), self.params, cv=4, scoring='roc_auc', n_jobs=n_jobs, verbose=self.verbose)

        self.best_params_ = None

    def fit(self, X, y=None):
        X = check_array(X)
        self.random_state_ = check_random_state(self.random_state)

        n_samples, n_features = X.shape

        # search hyperparameters
        n_noise = n_samples
        noise = _create_noise(n_noise, n_features)
        X_i = _merge(X, noise)
        y_i = _get_labels(n_samples, X_i.shape[0])
        self.rndsearch_clf.fit(X_i, y_i)
        self.best_params_ = self.rndsearch_clf.best_params_
        return self

    def transform(self, X):
        # Check if fit had been called
        check_is_fitted(self, 'random_state_')

        # Input validation
        X = check_array(X, accept_sparse=False)

        if self.outlier_inference:
            # add no noise when detect outliers
            return X

        n_samples, n_features = X.shape
        n_noise = n_samples
        full_noise = []

        # create noise
        clf = RandomForestClassifier(**self.best_params_)
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

