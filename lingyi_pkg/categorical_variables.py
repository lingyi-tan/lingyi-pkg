import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABCMeta, abstractmethod

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from lingyi_pkg.plot_helper import plot_roc, plot_pr


class Simulator(metaclass=ABCMeta):
    def __init__(self,
                 n_level_min=5,
                 n_level_max=10,
                 n_sample=1000,
                 n_feature=4,
                 feature_corr=0.7,
                 n_noise=0,
                 noise_magnitude=0.3,
                 positive_ratio=0.4,
                 random_state=None,
                 multiplicative_feature=True):

        self.n_level_min = n_level_min
        self.n_level_max = n_level_max
        self.n_sample = n_sample
        self.n_feature = n_feature
        self.feature_corr = feature_corr
        self.n_noise = n_noise
        self.noise_magnitude = noise_magnitude
        self.positive_ratio = positive_ratio
        self.random_state = random_state
        self.multiplicative_feature = multiplicative_feature

        self.feature_levels = None
        self.scores = None
        self.noised_scores = None
        self.target = None

        return

    def generate_feature_levels(self):
        feature_levels = {}
        if self.random_state is not None:
            np.random.seed(self.random_state)

        for i in range(self.n_feature + self.n_noise):
            n_levels = np.random.randint(low=self.n_level_min, high=self.n_level_max)
            levels = np.random.randint(low=0, high=n_levels, size=self.n_sample)
            feature_levels[f'feature_{i}'] = levels

        self.feature_levels = pd.DataFrame.from_dict(feature_levels)

        return self.feature_levels

    def _level2score(self, feature_levels):
        def func(i, levels):
            np.random.seed(i)
            level_score = np.random.rand(max(levels) + 1)
            return level_score[levels]

        scores_tb = pd.DataFrame.from_records(
            [func(int(idx[-1]), levels.values) for idx, levels in feature_levels.iteritems()]).T

        if self.multiplicative_feature:
            log_idx = np.random.choice(scores_tb.columns.values, size = int(scores_tb.shape[1]/2), replace=False)
            scores_tb[log_idx] = np.log(scores_tb[log_idx])
        return scores_tb.sum(axis=1)

    def generate_noised_scores(self):
        assert self.feature_levels is not None, "please call 'generate_feature_levels' first"
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.scores = self._level2score(self.feature_levels.iloc[:, np.arange(self.n_feature)])
        noise = np.random.normal(loc=0, scale=self.scores.std() * 3 * self.noise_magnitude, size=self.n_sample)
        self.noised_scores = self.scores + noise

        return self.noised_scores

    def generate_target(self):
        assert self.feature_levels is not None, "please call 'generate_feature_levels' first"
        assert self.noised_scores is not None, "please call 'generate_noised_scores' first"
        self.target = (self.noised_scores >= np.quantile(a=self.noised_scores, q=1.0 - self.positive_ratio)).astype(int)
        return self.target

    def reset(self):
        self.feature_levels = None
        self.scores = None
        self.noised_scores = None
        self.target = None
        return

    def sample_per_unique_value(self):
        assert self.feature_levels is not None, "please call 'generate_feature_levels' first"
        n_unique = self.feature_levels.drop_duplicates().shape[0]
        return self.n_sample/n_unique

    def plot_score_density(self, use_noised=True):
        scores = self.noised_scores if use_noised else self.scores
        plot_df = pd.DataFrame({'target': self.target, 'score': scores})
        sns.displot(plot_df, x='score', hue='target', kind='kde')
        return

    def plot_score_noise(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        sns.kdeplot(x=self.scores, ax=ax[0])
        sns.scatterplot(x=self.scores, y=self.noised_scores, marker='.', alpha=0.5)
        return
