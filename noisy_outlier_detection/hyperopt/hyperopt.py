from typing import Type

import numpy as np
from scipy.stats.distributions import uniform, randint
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

from ..model import NoisyOutlierDetector


class HyperparameterOptimizer(RandomizedSearchCV):

    def __init__(self, estimator: Type[NoisyOutlierDetector], n_estimators=(50, 150),
                 ccp_alpha=(0.01, 0.3), min_samples_leaf=(5, 100), n_sampling_iter: int = 20,
                 contamination: float = 0.05, n_trials: int = 30, n_cross_validations: int = 4, n_jobs: int = None,
                 random_state: int = None, verbose: bool = True):
        """
        Performes a randomized search to find good hyperparameters for the NoisyOutlierDetector. See paper for details.
        Usage is as follows:



        :param estimator:
        :param n_estimators:
        :param ccp_alpha:
        :param min_samples_leaf:
        :param n_sampling_iter:
        :param contamination:
        :param n_trials:
        :param n_cross_validations:
        :param n_jobs:
        :param random_state:
        :param verbose:
        """

        self.range_n_estimators = n_estimators
        self.range_ccp_alpha = ccp_alpha
        self.range_min_samples_leaf = min_samples_leaf
        self.contamination = contamination
        self.n_trials = n_trials
        self.cv = n_cross_validations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.param_grid = {
            'n_estimators': randint(*self.range_n_estimators),
            'ccp_alpha': uniform(*self.range_ccp_alpha),
            'min_samples_leaf': randint(*self.range_min_samples_leaf),
            'n_sampling_iter': [n_sampling_iter],
            'n_jobs': [None],
        }

        super().__init__(
            estimator=estimator(),
            param_distributions=self.param_grid,
            n_iter=n_trials,
            scoring=metrics.make_scorer(self.percentile_scoring, needs_proba=True),
            cv=self.cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    def percentile_scoring(self, y, y_pred):
        percentile_score = np.percentile(y_pred, 100 - (self.contamination * 100))
        minimal_score = min(y_pred)
        score = percentile_score - minimal_score
        return score

    def fit(self, X):
        """
        Same as optimizer.optimize(X). Use fit for compatibility to scikit-learn.
        :param X: data
        :return: optimized model for outlier detection
        """
        y = np.zeros(X.shape[0])
        return super(HyperparameterOptimizer, self).fit(X, y)

    def predict_outlier_probability(self, X):
        return self.predict_proba(X)[:, 1]

    def decision_function(self, X):
        return self.predict_outlier_probability(X)
