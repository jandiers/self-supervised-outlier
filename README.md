# Self-Supervised Learning for Outlier Detection

The detection of outliers can be very challenging, especially if the data has features that do not carry 
information about the outlyingness of a point.For supervised problems, there are many methods for selecting 
appropriate features. For unsupervised problems it can be challenging to select features that are meaningful 
for outlier detection. We propose a method to transform the unsupervised problem of outlier detection into a 
supervised problem to mitigate the problem of irrelevant features and the hiding of outliers in these features. 
We benchmark our model against common outlier detection models and have clear advantages in outlier detection 
when many irrelevant features are present.

This repository contains the code used for the experiments, as well as instructions to reproduce our results. 
For reproduction of our results, please switch to the "publication" branch 
or click [here](https://github.com/JanDiers/self-supervised-outlier/tree/publication).

As soon as our paper will be published online, the link for interested readers will appear here.

## Installation

The software can be installed by using pip. We recommend to use a virtual environment for installation, for example 
venv. [See the official guide](https://docs.python.org/3/library/venv.html).

To install our software, run

``pip install noisy_outlier``


## Usage

For outlier detection, you can use the `NoisyOutlierDetector` as follows. The methods follow the scikit-learn syntax:

```python
import numpy as np
from noisy_outlier import NoisyOutlierDetector
X = np.random.randn(50, 2)  # some sample data
model = NoisyOutlierDetector()
model.fit(X)
model.predict(X)  # returns binary decisions, 1 for outlier, 0 for inlier
model.predict_outlier_probability(X)  # predicts probability for being an outlier, this is the recommended way   
```

The `NoisyOutlierDetector` has several hyperpararameters such as the number of estimators for the classification 
problem or the pruning parameter. To our experience, the default values for the `NoisyOutlierDetector` provide stable 
results. However, you also have the choice to run routines for optimizing hyperparameters based on a RandomSearch. Details
can be found in the paper. Use the `HyperparameterOptimizer` as follows:

````python
import numpy as np
from scipy.stats.distributions import uniform, randint
from sklearn import metrics

from noisy_outlier import HyperparameterOptimizer, PercentileScoring
from noisy_outlier import NoisyOutlierDetector

X = np.random.randn(50, 5)
grid = dict(n_estimators=randint(50, 150), ccp_alpha=uniform(0.01, 0.3), min_samples_leaf=randint(5, 10))
optimizer = HyperparameterOptimizer(
                estimator=NoisyOutlierDetector(),
                param_distributions=grid,
                scoring=metrics.make_scorer(PercentileScoring(0.05), needs_proba=True),
                n_jobs=None,
                n_iter=5,
                cv=3,
            )
optimizer.fit(X)
# The optimizer is itself a `NoisyOutlierDetector`, so you can use it in the same way:
outlier_probability = optimizer.predict_outlier_probability(X)
````
Details about the algorithms may be found in our publication. 
If you use this work for your publication, please cite as follows. To reproduce our results, 
please switch to the "publication" branch or click [here](https://github.com/JanDiers/self-supervised-outlier/tree/publication).

````
Diers, J, Pigorsch, C. Self‐supervised learning for outlier detection. Stat. 2021; 10e322. https://doi.org/10.1002/sta4.322 
````

BibTeX:

````
@article{https://doi.org/10.1002/sta4.322,
author = {Diers, Jan and Pigorsch, Christian},
title = {Self-supervised learning for outlier detection},
journal = {Stat},
volume = {10},
number = {1},
pages = {e322},
keywords = {hyperparameter, machine learning, noisy signal, outlier detection, self-supervised learning},
doi = {https://doi.org/10.1002/sta4.322},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.322},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/sta4.322},
note = {e322 sta4.322},
year = {2021}
}
````
