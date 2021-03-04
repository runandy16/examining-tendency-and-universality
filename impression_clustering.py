# coding:utf-8

import os
import numpy as np
import scipy.spatial.distance as sdist
import scipy.cluster.hierarchy as hierarchy
from collections import OrderedDict
from tqdm import tqdm
from collections import Counter
from gensim.models import word2vec
from my_packages import my_functions
from matplotlib import rc
import matplotlib.pyplot as plt

font = {'family': 'IPAGothic'}
rc('font', **font)

class EstimateSimilarities(object):
    """評価語間の類似度を推定"""

    def __init__(self, threshold, data_dir, min_count, min_, iteration, size, window, hs, sg, num_top_words,
                 num_clusters=None):
        self.min_count = min_count
        self.min = min_  # ベクトル化の入力の最低出現確率
        self.iteration = iteration  # ベクトル化の反復数
        self.size = size
        self.window = window
        self.hs = hs  # 1なら階層ソフトマックス，0ならネガティブサンプリング
        self.sg = sg  # 1ならskip-gram，0ならCBOW
        self.data_dir = data_dir
        self.num_top_words = num_top_words
        self.num_clusters = num_clusters
        self.output_dir = 'impression_cluster'

        database = my_functions.load_json('outputs/parse_data/fashion/database_review.json')

        self.count_link_all = database['count_link_direct']
        self.reviews_list_all = database['texts_list_all']
        self.count_emotion_comb = database['count_word_emotion_comb']
        self.count_link_all = database['count_link_all']
        self.count_emotion_category = database['count_emotion_category']
        self.count_impression2emotion = database['count_impression2emotion']
        self.fashion_texts = database['texts_list_impression']

        self.impression_words = Counter(my_functions.flatten(self.fashion_texts))
        self.data = my_functions.load_json('impressions_cluster/level_cluster_words.json')

        self.cos_similarities = {}  # 評価語間の類似度を格納
        self.distances = []  # 評価語間の距離を格納
        self.count_words = {}

        self.model = None
        self.linkage = None
        self.threshold = threshold
        self.cluster_noun = []
        self.impressions_distances = [{}]

    def run(self):
        """評価語間の類似度を推定"""

        for k, v in self.data.items():
            self.removed_impression_words = v
            self.name = k
            self.estimate_similarities()  # 評価語間の類似度を推定
            self.linkage_threshold = self.linkage[-(int(self.num_clusters) - 1), 2]

            self.output_linkage()  # 階層型クラスタリングの可視化
            self.classify_impressions()
            self.calculate_gravities()

            self.output_similarities()  # 類似度行列の出力と保存

    def estimate_similarities(self):
        """評価語間の類似度を推定"""

        if not os.path.isdir(self.data_dir + '/word2vec_models'):
            os.mkdir(self.data_dir + '/word2vec_models')
        os.chdir(self.data_dir + '/word2vec_models')

        self.distances = [[0.0 for _ in range(len(self.removed_impression_words))]
                          for _ in range(len(self.removed_impression_words))]

        for i in tqdm(range(self.iteration), desc='conduct word2vec'):
            model_name = 'word2vec_{}'.format(i)  # モデルの名称

            if not os.path.isfile(model_name):  # データが無い場合モデル作成
                self.model = word2vec.Word2Vec(self.reviews_list_all,
                                               size=self.size,
                                               min_count=self.min,  # min_count以下の単語は削除済みなので0
                                               workers=5,
                                               window=self.window,
                                               hs=self.hs,  # 1:hierarchical softmax, 0:negative sampling
                                               sg=self.sg)  # 1:skip-gram, 0: CBOW
                self.model.save(model_name)  # word2vec1試行の保存
            else:  # モデル読み込み
                self.model = word2vec.Word2Vec.load(model_name)
            # 評価語間の類似度を保持
            for j, w1 in enumerate(self.removed_impression_words):

                self.cos_similarities['{} {}'.format(w1, w1)] = \
                    self.cos_similarities.get('{} {}'.format(w1, w1), 0.0) + 1.0

                for k, w2 in enumerate(self.removed_impression_words[j + 1:]):
                    cos_similarity = self.model.wv.similarity(w1, w2)
                    distance = 1 - cos_similarity

                    self.cos_similarities['{} {}'.format(w1, w2)] = \
                        self.cos_similarities.get('{} {}'.format(w1, w2), 0.0) + cos_similarity
                    self.cos_similarities['{} {}'.format(w2, w1)] = self.cos_similarities['{} {}'.format(w1, w2)]

                    self.distances[j][k + j + 1] += distance
                    self.distances[k + j + 1][j] += distance

        self.cos_similarities = {key: cos_similarity / self.iteration
                                 for key, cos_similarity in list(self.cos_similarities.items())}

        for j in range(len(self.removed_impression_words)):  # word2vec試行の平均をとる
            for k in range(len(self.removed_impression_words[j + 1:])):
                self.distances[j][k + j + 1] /= self.iteration
                self.distances[k + j + 1][j] /= self.iteration

        self.cos_distance = np.percentile(my_functions.flatten(self.distances), self.threshold)

        plt.hist(my_functions.flatten(self.distances), bins=50)
        self.linkage = hierarchy.linkage(sdist.squareform(np.array(self.distances)), method='complete', metric='cosine')

    def create_labels(self):

        return OrderedDict([(word, 'k') for word in self.removed_impression_words])

    def output_linkage(self):
        """階層型クラスタリングの可視化"""

        labels = self.create_labels()

        dpi = 200
        height = int(len(self.removed_impression_words) / 8)
        if dpi * height > 32767:
            height = int(32000 / 200)
        plt.figure(num=None, figsize=(5, height), dpi=dpi, facecolor='w', edgecolor='b')
        hierarchy.dendrogram(self.linkage, color_threshold=self.cos_distance, orientation='right',
                             labels=np.array(list(labels.keys())))
        plt.vlines(x=self.cos_distance, ymin=0, ymax=len(self.removed_impression_words) * 10, linestyle=":")
        for y_label in plt.gca().get_ymajorticklabels():
            y_label.set_color(labels[y_label.get_text()])

        plt.savefig(self.output_dir + f'/linkage_impression_{self.name}.png', bbox_inches='tight')

    def classify_impressions(self):

        fcluster = hierarchy.fcluster(self.linkage, self.cos_distance, criterion='distance')
        self.cluster_impression = [[] for _ in list(set(fcluster.tolist()))]
        for fc, impression_word in zip(fcluster, self.removed_impression_words):
            self.cluster_impression[fc - 1].append(impression_word)

    def calculate_gravities(self):

        self.cluster_vec = [[] for _ in self.cluster_impression]
        self.cluster_grav = [{} for _ in self.cluster_impression]
        self.impression_score = [{} for _ in self.cluster_impression]
        self.gravs = []

        for i, cluster in enumerate(self.cluster_impression):
            for word in cluster:
                self.cluster_vec[i].append(self.model.wv[word])

        for i, (vecs, cluster) in tqdm(enumerate(zip(self.cluster_vec, self.cluster_impression)),
                                       desc=f'calculate_gravities{self.name}'):
            grav = np.mean(vecs, axis=0)
            self.gravs.append(grav)
            for j, vec in enumerate(vecs):
                self.cluster_grav[i].setdefault(cluster[j], cos_sim(grav, vec))

        for k, vec in enumerate(self.gravs):
            for impression_word, cat_count in list(self.count_impression2emotion.items()):
                try:
                    self.impression_score[k].setdefault(impression_word, cos_sim(self.model.wv[impression_word], vec))
                except:
                    continue

    def output_similarities(self):
        """クラスタリング結果の出力"""
        my_functions.output_json({'cluster_impression': self.cluster_impression},
                                 f'cluster_impression_{self.name}.json',
                                 self.output_dir)
        sort_dic = {}
        for list in self.count_link_all:
            if list[0] not in sort_dic.keys():
                sort_dic[list[0]] = {}
                sort_dic[list[0]].setdefault(list[1], int(list[2]))
            else:
                sort_dic[list[0]].setdefault(list[1], int(list[2]))

        duplicate_words = []
        with open(self.output_dir + f'/classified_words_{self.name}.tsv', 'w') as f:
            for i, impression_words in enumerate(self.cluster_impression):
                f.write('-- cluster {}\n'.format(i + 1))
                for impression_word in impression_words:
                    f.write("{}\t".format(impression_word))
                    f.write(', '.join(
                        [word[0] for word in sorted(sort_dic[impression_word].items())[:10] if word[0] != '-']) + '\n')
                f.write("\n")


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def test():
    data_dir = 'outputs/parse_data/fashion'

    es = EstimateSimilarities(threshold=40,  # cos距離を区切る割合
                              data_dir=data_dir,
                              min_count=0.01,
                              min_=5,
                              iteration=100,  # 100
                              size=200,
                              window=5,
                              hs=1,
                              sg=1,
                              num_top_words=0,
                              num_clusters=50)
    es.run()


if __name__ == '__main__':
    test()
