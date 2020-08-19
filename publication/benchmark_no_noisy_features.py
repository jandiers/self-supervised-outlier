from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import precision_n_scores
from scipy.stats.distributions import loguniform, uniform, randint
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import ParameterSampler

from noisy_outlier_detection import NoisyOutlierDetector, HyperparameterOptimizer, PercentileScoring
from publication.data_loading import ArffFileLoader

files = [
    'Lymphography_withoutdupl_norm_idf.arff',
    'WPBC_withoutdupl_norm.arff',
    'HeartDisease_withoutdupl_norm_05_v10.arff',
    'Stamps_withoutdupl_norm_05_v10.arff',
]


def build_ae_hidden_layers(n_features, n_layers):
    first_hidden = n_features * 0.8
    emb_dim = max(1, n_features // 5)
    layers = np.linspace(first_hidden, emb_dim, num=n_layers).astype(int)
    layers = np.hstack((layers, layers[::-1][1:]))
    return list(layers)


for file in files:
    path = Path().cwd()
    loader = ArffFileLoader(name=file.split('_')[0], path=path.joinpath('data').joinpath(file).__str__())
    X, y = loader.load()

    n_samples, n_features = X.shape

    param_distributions = {
        'NoisyOutlierDetector': {'n_estimators': range(100, 150), 'n_jobs': [None]},
        'KNN': {'n_neighbors': range(2, min(min(100, n_samples), 100)), 'contamination': [0.05]},
        'LOF': {'n_neighbors': range(2, min(min(100, n_samples), 100)), 'contamination': [0.05]},
        'IForest': {'n_estimators': range(2, 5 * n_features), 'random_state': [28], 'contamination': [0.05]},
        'OCSVM': {'gamma': loguniform(0.0000001, 1), 'contamination': [0.05]},
        'AutoEncoder': {'hidden_neurons': [build_ae_hidden_layers(n_features, n_layers) for n_layers in range(1, 15)]},
    }

    grid = dict(n_estimators=randint(50, 150), ccp_alpha=uniform(0.01, 0.3), min_samples_leaf=randint(5, 20))
    our_alg = HyperparameterOptimizer(NoisyOutlierDetector(), param_distributions=grid,
                                      scoring=make_scorer(PercentileScoring(0.05), needs_proba=True),
                                      n_jobs=4,
                                      n_iter=5,
                                      cv=3,
                                      )
    our_alg.decision_function = our_alg.predict_outlier_probability
    our_alg.__name__ = 'NoisyOutlierDetector'

    algorithms = [our_alg, KNN, LOF, IForest, OCSVM, AutoEncoder]

    result = defaultdict(list)

    for alg in algorithms:

        alg_name = alg.__name__

        distribution = param_distributions[alg_name]
        dist_combination = list(ParameterSampler(distribution, n_iter=10, random_state=28))

        for dc in tqdm.tqdm(dist_combination, desc=f'Processing {alg_name} on {file}'):

            if alg_name == 'NoisyOutlierDetector':
                clf = alg.fit(X)
            else:
                clf = alg(**dc).fit(X)

            dc = dict(((alg_name, key), value) for (key, value) in dc.items())

            for meth_name, meth in zip(['dec_func', 'predict'], [clf.decision_function, clf.predict]):

                y_pred = meth(X)
                auc = roc_auc_score(y, y_pred).round(4)
                dc.update({(alg_name, f'roc_auc_{meth_name}'): auc})

                if meth_name == 'dec_func':
                    patn = precision_n_scores(y, y_pred).round(4)
                    dc.update({(alg_name, f'precision@n_{meth_name}'): patn})

            for k, v in dc.items():
                result[k].append(v)

    df = pd.DataFrame(result)

    dataname = file.split('.')[0]

    """
    # for immediate plots of results
    for m in ['roc_auc_dec_func', 'roc_auc_predict', 'precision@n_dec_func']:
        df.xs(m, level=1, axis=1).plot.box(title=f'{dataname}, {m}')
        plt.savefig(path.joinpath(f'./results/{m}_{dataname}.pdf'), dpi=300)
        plt.show()
    """

    df.to_csv(path.joinpath(f'./results/{dataname}.csv'))
