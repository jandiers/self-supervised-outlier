from sklearn import metrics
from sklearn.preprocessing import minmax_scale
from noisy_outlier import NoisyOutlierDetector, HyperparameterOptimizer, PercentileScoring
from scipy.stats.distributions import uniform, randint
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.utility import precision_n_scores
from scipy.stats import uniform, norm
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from publication.data_loading import ArffFileLoader

sns.set_context('paper')

n_repetitions = 10  # repeat experiments to reduce randomness

files = [
    'Lymphography_withoutdupl_norm_idf.arff',
    'Stamps_withoutdupl_norm_05_v10.arff',
    'HeartDisease_withoutdupl_norm_05_v10.arff',
    'WPBC_withoutdupl_norm.arff',
]

NoisyOutlierDetector.decision_function = NoisyOutlierDetector.predict_outlier_probability

stamps_models = [
    ('Ours', NoisyOutlierDetector(n_jobs=4)),
    ('KNN', KNN(n_neighbors=72)),
    ('LOF', LOF(n_neighbors=100)),
    ('IForest', IForest(n_estimators=29)),
    ('OCSVM', OCSVM(gamma=1.9e-06)),
    ('AutoEncoder', AutoEncoder(hidden_neurons=[7, 4, 1, 4, 7]))
]

heartdisease_models = [
    ('Ours', NoisyOutlierDetector(n_jobs=4)),
    ('KNN', KNN(n_neighbors=45)),
    ('LOF', LOF(n_neighbors=90)),
    ('IForest', IForest(n_estimators=60)),
    ('OCSVM', OCSVM(gamma=7.494e-07)),
    ('AutoEncoder', AutoEncoder(hidden_neurons=[10, 8, 6, 4, 2, 4, 6, 8, 10])),
]

wpbc_models = [
    ('Ours', NoisyOutlierDetector(n_jobs=4)),
    ('KNN', KNN(n_neighbors=12)),
    ('LOF', LOF(n_neighbors=18)),
    ('IForest', IForest(n_estimators=162)),
    ('OCSVM', OCSVM(gamma=7.494e-07)),
    ('AutoEncoder', AutoEncoder(hidden_neurons=[26, 19, 12, 6, 12, 19, 26])),
]

lympho_models = [
    ('Ours', NoisyOutlierDetector(n_jobs=4)),
    ('KNN', KNN(n_neighbors=14)),
    ('LOF', LOF(n_neighbors=62)),
    ('IForest', IForest(n_estimators=41)),
    ('OCSVM', OCSVM(gamma=4.87e-01)),
    ('AutoEncoder', AutoEncoder(hidden_neurons=[14, 8, 3, 8, 14])),
]

model_collection = {
    'Stamps': stamps_models,
    'HeartDisease': heartdisease_models,
    'WPBC': wpbc_models,
    'Lymphography': lympho_models,
}

for file in files:
    path = Path().cwd()
    loader = ArffFileLoader(name=file.split('_')[0], path=path.joinpath('data').joinpath(file).__str__())
    X, y = loader.load()

    models = model_collection[file.split('_')[0]]
    n_samples = X.shape[0]

    results = []

    for repet in range(n_repetitions):
        for n_noise_features in tqdm(range(0, 101, 2)):
            X_loop = np.hstack([
                X,
                norm().rvs((n_samples, n_noise_features)),
                uniform().rvs((n_samples, n_noise_features))
            ])

            n_noise_features = 2 * n_noise_features  # double because one uniform and normal feature has been added

            X_loop = minmax_scale(X_loop)

            # search for hyperparameter for our model, since the others already have fixed best hyperparameters
            if n_noise_features == 0:
                grid = dict(n_estimators=randint(50, 150), ccp_alpha=uniform(0.01, 0.3),
                            min_samples_leaf=randint(5, 10))
                hyp_model = HyperparameterOptimizer(estimator=NoisyOutlierDetector(n_jobs=4),
                                                    param_distributions=grid,
                                                    scoring=metrics.make_scorer(PercentileScoring(0.05),
                                                                                needs_proba=True),
                                                    n_jobs=4,
                                                    n_iter=5,
                                                    cv=3, )
                hyp_model.fit(X)

            for name, m in models:
                if name == 'Ours':
                    m = hyp_model.best_estimator_

                m.fit(X_loop)
                scores = m.decision_function(X_loop)

                auc = metrics.roc_auc_score(y, scores)
                patn = precision_n_scores(y, scores).round(4)

                results.append({'repetition': repet, 'model': name, 'AUC': auc, 'PrecAtN': patn,
                                'n_noise_features': n_noise_features})

    df = pd.DataFrame(results)
    df = df.set_index(['n_noise_features', 'model', 'repetition']).unstack(level=1)

    df_auc = df['AUC'][['Ours', 'KNN', 'LOF', 'IForest', 'OCSVM', 'AutoEncoder']]

    df_auc.to_csv(path.joinpath(f'results/with_noisy_features/{file.split("_")[0]}.csv'))

    df_auc.rolling(5).mean().plot()

    df['AUC'].rolling(5).mean().plot(
        figsize=(12, 8),
        title='AUC value with increasing number of non-informative features'
    )

    plt.xlabel('Number non-informative features')
    _ = plt.ylabel('AUC value (smoothed)')

    plt.savefig(
        path.joinpath(f'results/with_noisy_features/{file.split(".")[0]}_{n_noise_features}_noise_features.pdf'),
        dpi=300)
