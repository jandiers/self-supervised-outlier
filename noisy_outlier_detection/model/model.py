import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from .transformer import IterativeNoiseTransformer, _get_labels


class _MyRandomForest(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        y = _get_labels(self.n_real_samples, X.shape[0])
        return super(_MyRandomForest, self).fit(X, y, sample_weight=sample_weight)

    def adjust_labels(self, data_size):
        self.n_real_samples = data_size


class NoisyOutlierDetector(Pipeline):

    def __init__(self, n_estimators: int = 100,
                 ccp_alpha: float = 0.01, n_sampling_iter: int = 20,
                 min_samples_leaf: int = 5,
                 verbose: bool = False,
                 n_jobs=None
                 ):
        """
        The NoisyOutlierDetector performs outlier detection based on self-supervised learning. It learns differences between
        the provided data and uniform noise. See paper for details. Usage is as follows:

        >>> import numpy as np
        >>> from noisy_outlier_detection import NoisyOutlierDetector
        >>> X = np.random.randn(50, 2)
        >>> model = NoisyOutlierDetector()
        >>> model.fit(X)
        >>> model.predict(X)  # returns binary decisions
        >>> model.predict_outlier_probability(X)  # predicts probability for being an outlier

        :param n_estimators: number of estimators used in Random Forest, defaults to 100
        :param ccp_alpha: pruning parameter, defaults to 0.01
        :param n_sampling_iter: number of iterations to draw uniform noise, defaults to 20
        :param min_samples_leaf: minimum number of samples at the leaf, defaults to 20
        :param verbose: if True, print progress messages, defaults to False
        :param n_jobs: number of kernels used for estimates. defaults to None, meaning no parallel processing
        """

        # set attributes
        self.n_estimators, self.ccp_alpha, \
            self.n_sampling_iter, self.min_samples_leaf, \
            self.verbose, self.n_jobs \
            = \
            n_estimators, ccp_alpha, n_sampling_iter, \
            min_samples_leaf, verbose, n_jobs

        # construct pipeline
        super(NoisyOutlierDetector, self).__init__([
            (
                'scaler', MinMaxScaler(
                    feature_range=(0, 1)
                )
            ),
            (
                'noise', IterativeNoiseTransformer(
                    n_sampling_iter=self.n_sampling_iter,
                    outlier_inference=False,
                    verbose=self.verbose,
                )
            ),
            (
                'clf', _MyRandomForest(
                    n_estimators=self.n_estimators,
                    min_samples_leaf=self.min_samples_leaf,
                    ccp_alpha=self.ccp_alpha,
                    class_weight='balanced',
                    oob_score=True,
                    verbose=self.verbose)
            )
        ]
        )

    def fit(self, X, y=None):
        """
        Fits the outlier detection on the data X.

        :param X: array-like, data to find outliers in
        :return: NoisyOutlierDetector
        """
        self._outlier_inference(False)
        self.named_steps['clf'].adjust_labels(X.shape[0])
        r = super(NoisyOutlierDetector, self).fit(X)
        self._outlier_inference(True)
        return r

    def _outlier_inference(self, detect_outliers: bool = True):
        self.named_steps['noise'].outlier_inference = detect_outliers

    def predict_outlier_probability(self, X: np.ndarray):
        """
        Predicts the probability for being an outlier.

        :param X: numpy array
        :return:
        """
        return self.predict_proba(X)[:, 1]

