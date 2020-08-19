import numpy as np
from sklearn.model_selection import RandomizedSearchCV


class PercentileScoring(object):
    def __init__(self, contamination: float = 0.05):
        """
        Metric to compute the range between maximum and minimum outlier score. Considers the 1-contamination quantile
        as maximum outlier score.

        :param contamination: expected contamination to optimize for
        """
        self.contamination = contamination

    def __call__(self, y, y_pred):
        percentile_score = np.percentile(y_pred, 100 - (self.contamination * 100))
        minimal_score = min(y_pred)
        score = percentile_score - minimal_score
        return score


class HyperparameterOptimizer(RandomizedSearchCV):
    """
    Searches for suitable hyperparameters for the noisy outlier detection. See paper for details.
    Usage is as follows:

    >>> import numpy as np
    >>> from scipy.stats.distributions import uniform, randint
    >>> from sklearn import metrics
    >>> from noisy_outlier import HyperparameterOptimizer, PercentileScoring
    >>> from noisy_outlier import NoisyOutlierDetector
    >>> X = np.random.randn(50, 5)
    >>> grid = dict(n_estimators=randint(50, 150), ccp_alpha=uniform(0.01, 0.3), min_samples_leaf=randint(5, 10))
    >>> optimizer = HyperparameterOptimizer(
    >>>                         estimator=NoisyOutlierDetector(),
    >>>                         param_distributions=grid,
    >>>                         scoring=metrics.make_scorer(PercentileScoring(0.05), needs_proba=True),
    >>>                         n_jobs=None,
    >>>                         n_iter=5,
    >>>                         cv=3,
    >>>                         )
    >>> optimizer.fit(X)
    """

    def fit(self, X):
        y = np.zeros(X.shape[0])
        return super(HyperparameterOptimizer, self).fit(X, y)

    def predict_outlier_probability(self, X):
        return self.predict_proba(X)[:, 1]
