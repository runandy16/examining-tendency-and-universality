import pandas as pd
import numpy as np
import statsmodels.api as sm
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from sklearn.cluster import KMeans
from matplotlib import rc

try:
    from my_packages import my_functions
except ImportError:
    import my_functions


font = {'family': 'IPAGothic'}
rc('font', **font)


def kmeans_aic(model, X):
    k, m = model.cluster_centers_.shape
    if isinstance(X, xr.DataArray):
        n = X.flat.values.shape[0]
    else:
        n = X.shape[0]
    d = model.inertia_
    aic = d + 2 * m * k
    delattr(model, 'labels_')
    return aic


class TimeAnalysis(object):
    def __init__(self,level):
        self.database = my_functions.load_json('outputs/database_vecs.json')
        self.impression_words = self.database['impression_words']
        self.season_vecs = self.database['season_vecs']
        self.fashion_times = self.database['fashion_times']

        self.level = level
        self.new_season_vecs = []

    def run(self):
        self.state_space_model()
        n_cluster = self.tsclusteringN()
        km = KMeans(n_clusters=n_cluster, n_init=100).fit(np.array(self.new_season_vecs))
        self.plot_clustering(km, n_cluster)

    def state_space_model(self):
        """状態空間モデルの推定"""
        for word, vec in zip(self.impression_words, self.season_vecs):
            endog = pd.Series(vec, index=self.fashion_times)
            mod_local_level = sm.tsa.UnobservedComponents(endog=endog, level='lldtrend', seasonal=12)

            res_trend = mod_local_level.fit(method='bfgs')  # BFGS法で最適化する

            if self.level == 'level':
                self.new_season_vecs.append(res_trend.level['smoothed'].tolist())  # 内部状態のリスト作成
            else:
                self.new_season_vecs.append(res_trend.seasonal['smoothed'].tolist())  # 季節状態のリスト作成

            rcParams['figure.figsize'] = 15, 20
            res_trend.plot_components().savefig(f'impressions_trend/{word}_trend.png')

    def tsclusteringN(self):
        """クラスタ数を定める"""
        n_clusters = [n for n in range(2, 30)]
        aics = []
        for n in n_clusters:
            km = KMeans(n_clusters=n, n_init=100).fit(np.array(self.new_season_vecs))
            AIC = kmeans_aic(km, np.array(self.new_season_vecs))
            aics.append(AIC)  # AICによってクラスタ数を定める
        min_value = min(aics)
        min_index = aics.index(min_value)
        # AICをプロット
        fig = plt.figure(figsize=(20, 15))
        plt.plot(n_clusters, aics, 'r-o')
        plt.title("AIC")
        plt.xlabel("clusters")
        plt.savefig('impressions_trend/clusters_aic.png')

        return min_index + 2

    def plot_clustering(self, km, n_clusters):

        plt.figure(figsize=(20, 15))

        # クラスターごとの中心をプロット
        for i, c in enumerate(km.cluster_centers_):
            plt.plot(c, label=f'pattern {i + 1}')
        plt.xticks([i for i in range(len(self.fashion_times))], self.fashion_times, rotation=50)
        plt.title(f"{self.level} center")
        plt.xlabel("age")
        plt.legend(loc='center left', fontsize=20)
        plt.savefig(f'impressions_cluster/{self.level}_cluster_center.png')
        plt.clf()

        # クラスターごとのプロット
        cluster_words = {i: [] for i in range(n_clusters)}
        for i in range(n_clusters):
            plt.figure(figsize=(20, 10))
            plt.title(f'pattern{i + 1}', fontdict={"fontsize": 18, "fontweight": "bold"})

            for label, d, t in zip(km.labels_, np.array(self.new_season_vecs), self.impression_words):
                if label == i:
                    cluster_words[i].append(t)
                    plt.plot(d, label=t)

            plt.legend(borderaxespad=0, ncol=2, fontsize=8)
            plt.xticks([i for i in range(len(self.fashion_times))], self.fashion_times, rotation=50)
            plt.subplots_adjust(left=0.2, right=0.6)
            plt.savefig(f'impressions_cluster/{self.level}_cluster_labeled{i}.png')
            plt.clf()

        my_functions.output_json(cluster_words, f'impressions_cluster/{self.level}_cluster_words.json')


if __name__ == '__main__':
    test = TimeAnalysis(level='level')
    test.run()
