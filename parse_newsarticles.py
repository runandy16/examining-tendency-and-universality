import os
import codecs
import xlrd
import re
import operator
import copy
import sys
import subprocess
import emoji
import multiprocessing
import neologdn

from pyknp import Juman
from tqdm import tqdm
from collections import OrderedDict

try:
    from my_packages import my_functions
except ImportError:
    import my_functions


def jumanpp_analysis(word):
    """JUMAN++で形態素解析"""

    jumanpp = Juman()

    try:
        return jumanpp.analysis(word).mrph_list()
    except Exception as e:
        print(e)
        print(word)
        sys.exit()


def remove_emoji(src_str):
    return ''.join(c for c in src_str if c not in emoji.UNICODE_EMOJI)


def japanese_decision(word):
    if re.match('[ぁ-んァ-ン一-龥]', word):
        return True
    else:
        return False


def subprocess_analysis(text):
    """コマンドでの構文解析処理"""

    sentences = re.split('。|\.|．|\!|\?|！|？|\(|\)|（|）|【|】|☆|…|♪|&|#|;|:|◎|※', text)  # 文ごとにレビューをカット

    knp_sentences = []
    for sentence in sentences:  # 一文の解析
        sentence = neologdn.normalize(sentence)
        sentence = remove_emoji(sentence)
        sentence = ''.join(sentence.split()).strip()
        sentence = re.sub(re.compile("[!-/:-@[-`{-~]"), "", sentence)  # 半角記号を除去

        if not sentence:
            continue

        cmd = f"echo \"{sentence}\" | jumanpp | knp -simple -anaphora"
        retcode = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = retcode.communicate()
        try:
            knp_result = knp_utils(stdout_data.decode())
            knp_sentences.append(knp_result)
        except:
            continue

    return knp_sentences


def knp_utils(knp_result):
    """KNPのコマンド出力-simpleをディクショナリに整理"""
    result_dict = OrderedDict()
    for line in knp_result.split('\n'):

        line_info = line.split(" ")
        if line in {"EOS", ''}:
            break
        elif line.startswith('#'):
            result_dict.setdefault("S-ID", int(line_info[1].replace("S-ID:", "")))
            result_dict.setdefault("KNP", line_info[2].replace("KNP:", ""))
            result_dict.setdefault("DATE", line_info[3].replace("DATE:", ""))
            result_dict.setdefault('bnst', [])
        elif line.startswith('*'):  # 文節で区切る（基本句だと'+'）
            result_dict['bnst'].append({'parent_id': int(line_info[1][:-1]), 'mrph_list': []})
        elif line.startswith('+'):
            pass
        elif len(line_info) < 3:
            pass
        else:
            repnames = [line_info[2]]
            antonyms = []
            content_language = False
            if line_info[11] != 'NIL':
                imis = line.split('\"')[1].split(' ')
                for imi in imis:
                    if imi.startswith('代表表記:'):
                        repnames = imi.split(':')[1].split('/')
                    elif imi.startswith('反義:'):
                        antonyms = imi.split(':')[2].split('/')
                    elif imi.startswith('内容語'):
                        content_language = True

            result_dict['bnst'][-1]['mrph_list'].append(OrderedDict({'midasi': line_info[0],
                                                                     'yomi': line_info[1],
                                                                     'genkei': line_info[2],
                                                                     'hinsi1': line_info[3],
                                                                     'hinsi1_id': int(line_info[4]),
                                                                     'hinsi2': line_info[5],
                                                                     'hinsi2_id': int(line_info[6]),
                                                                     'katuyou1': line_info[7],
                                                                     'katuyou1_id': int(line_info[8]),
                                                                     'katuyou2': line_info[9],
                                                                     'katuyou2_id': int(line_info[10]),
                                                                     'repnames': repnames,
                                                                     'antonyms': antonyms,
                                                                     'content_language': content_language}))

    return result_dict


def output_count_link(dictionary, output_file, directory):
    """カウントした辞書を出力"""

    sort_list = sorted(list(dictionary.items()), key=lambda x: x[1])
    sort_list.reverse()
    sort_list = [l[0].split(' ') + [l[1]] for l in sort_list]
    my_functions.output_tsv(sort_list, output_file, directory)

    return sort_list


class LoadDictionaryData(object):
    """評価語リストの読み込み"""

    def __init__(self, data_dir, represent_dir, dictionary_dir, load_flag):
        self.data_dir = data_dir
        self.represent_dir = represent_dir
        self.dictionary_dir = dictionary_dir
        self.load_flag = load_flag

        self.impression_dict = {}  # 印象語
        self.internalized_dict = {}  # 内評価の評価語
        self.emotion_dict = {}  # 感情語
        self.emotion_comb_dict = {}  # 感情語と内評価を合わせた評価語
        self.emotion_category_dict = {c: set() for c in ['喜', '怒', '哀', '怖', '恥', '好', '厭', '昂', '安', '驚']}
        self.internalized2emotion = {}

        self.represent_data = {}  # 代表表記を格納する辞書

    def run(self):
        """評価語リストの読み込み"""

        if self.load_flag:

            if os.path.isfile('/database_dictionary.json'):
                dictionary_database = my_functions.load_json('/database_dictionary.json')
            else:
                print('Error: Not found: database_dictionary.json')
                sys.exit()

            self.impression_dict = dictionary_database['impression_dict']
            self.internalized_dict = dictionary_database['internalized_dict']
            self.emotion_dict = dictionary_database['emotion_dict']
            self.emotion_category_dict = dictionary_database['emotion_category_dict']
            self.emotion_comb_dict = dictionary_database['emotion_comb_dict']
            print('評価表現辞書データの読み込み完了')
        else:
            print('評価表現辞書データを新規作成')

            represent_file = 'database_represent.json'
            if os.path.isfile(self.represent_dir + '/' + represent_file):
                self.represent_data = my_functions.load_json(self.represent_dir + '/' + represent_file)
            else:
                print('JUMANAPPの変換ファイルを新規作成')
                self.represent_data = {}

            self.load_dictionaries()

            my_functions.output_json(self.represent_data, represent_file, self.represent_dir)
            dictionary_data = {'impression_dict': self.impression_dict,
                               'internalized_dict': self.internalized_dict,
                               'emotion_dict': self.emotion_dict,
                               'emotion_category_dict': self.emotion_category_dict,
                               'emotion_comb_dict': self.emotion_comb_dict
                               }
            my_functions.output_json(dictionary_data, 'database_dictionary.json', self.data_dir)

        return self

    def load_dictionaries(self):
        """評価辞書の読み込み"""

        self.load_polarity_declinable_dict()  # 日本語極性辞書用言編の読み込み
        self.load_polarity_nouns_dict()  # 日本語極性辞書名詞編の読み込み
        self.load_evaluation_dict()  # 評価値表現辞書の読み込み
        self.load_emotion_data()  # 感情表現辞典の読み込み
        self.load_internalized2emotion()  # 内評価と感情カテゴリの関係性データの読み込み"
        self.load_jappraisal()  # JAppraisal辞書の読み込み
        self.integrate_evaluation_words()  # 読み込んだ評価語情報の統合

    def convert_represent_word(self, word):
        """JUMANの代表表記に変換"""

        if word in self.represent_data:
            represent_word = self.represent_data[word]
        else:
            result = jumanpp_analysis(word)

            if len(result) == 1:
                represent_word = result[0].repname.split('/')[0].strip()
                if not represent_word:
                    represent_word = result[0].genkei
            else:
                represent_word = word

            self.represent_data.setdefault(word, represent_word)

        return represent_word

    def load_polarity_declinable_dict(self):
        """日本語極性辞書用言編の読み込み"""

        polarity2id = {'ポジ': 'p', 'ネガ': 'n'}
        remove_words = {'だ', 'です', 'と', 'の'}

        with codecs.open(self.dictionary_dir + '/wago.121808_revised.txt', 'r', 'cp932') as f_read:
            for line in tqdm(f_read.readlines(), desc='load polarity_declinable'):

                categories, word = line.split('\t')  # カテゴリと評価語に分割
                split = word.rstrip().split(' ')  # スペースで分割

                if split[-1] in remove_words:  # 指定した語尾の単語を除去
                    split = split[:-1]

                word = ''.join(split).strip()
                polarity, category = categories.split('（')  # 極性とカテゴリに分割

                if not word:
                    continue
                word = self.convert_represent_word(word.lower())  # 表記ゆれの統一
                self.impression_dict.setdefault(word, [('日本語極性辞書(用言編)', polarity2id[polarity])])

    def load_polarity_nouns_dict(self):
        """日本語極性辞書名詞編の読み込み"""

        with codecs.open(self.dictionary_dir + '/pn.csv.m3.120408.trim.txt', 'r', 'cp932') as f_read:
            for line in tqdm(f_read.readlines(), desc='load polarity_nouns'):
                word, polarity, category = line.split('\t')

                if category.find('評価') == -1:
                    continue

                if polarity not in {'p', 'e', 'n'}:
                    polarity = 'e'

                words = word.split('・')  # 表記が２つある場合の処理
                for word in words:
                    word = self.convert_represent_word(word.lower().strip())

                    if word in self.impression_dict:
                        self.impression_dict[word].append(('日本語極性辞書(名詞編)', polarity))
                    else:
                        self.impression_dict.setdefault(word, [('日本語極性辞書(名詞編)', polarity)])

    def load_evaluation_dict(self):
        """評価値表現辞書の読み込み"""

        with codecs.open(self.dictionary_dir + '/EVALDIC_ver1.01.txt', 'r', 'utf8') as f_read:
            for line in tqdm(f_read.readlines(), desc='load evaluation'):
                word = self.convert_represent_word(line.split('\n')[0].strip().lower())

                if word in self.impression_dict:
                    self.impression_dict[word].append(('評価値表現辞書', 'None'))
                else:
                    self.impression_dict.setdefault(word, [('評価値表現辞書', 'None')])

    def load_emotion_data(self):
        """感情表現辞典の読み込み"""

        book = xlrd.open_workbook(self.dictionary_dir + '/感情表現辞典語句.xlsx')
        sheet1 = book.sheet_by_index(0)

        for row in tqdm(range(1, sheet1.nrows), desc='load Emotion'):
            category = sheet1.cell(row, 3).value.strip()  # 感情の分類を取得

            for word in sheet1.cell(row, 1).value.split('/'):  # 表記のゆれごとに
                if '(' in word:  # ()内以外の部分を取得
                    word = word[:word.index('(')] + word[word.index(')') + 1:]
                    # 氷(を感じる)だけ取れない

                word = self.convert_represent_word(word.strip().lower())

                if word in self.emotion_dict:
                    if category in self.emotion_dict[word]:
                        continue
                    self.emotion_dict[word].append(category)
                else:
                    self.emotion_dict.setdefault(word, [category])
                self.emotion_category_dict[category].add(word)

    def load_internalized2emotion(self):
        """内評価と感情カテゴリの関係性データの読み込み"""

        book = xlrd.open_workbook(self.dictionary_dir + '/internalized2emotion.xlsx')
        sheet1 = book.sheet_by_index(0)

        for row in range(sheet1.nrows):
            categories = sheet1.cell(row, 2).value.split('/')
            if categories == ['']:
                continue

            self.internalized2emotion.setdefault(tuple(map(int, sheet1.cell(row, 0).value.split('-'))),
                                                 categories)

    def load_jappraisal(self):
        """JAppraisal辞書の読み込み"""

        book = xlrd.open_workbook(self.dictionary_dir + '/JAppraisal1.2/jappraisal.xlsx')
        sheet1 = book.sheet_by_index(0)
        sheet2 = book.sheet_by_index(1)

        # feature名とIDの関係を格納--------------------------------------------------------------------
        feature_id_dict = {}  # feature名とIDの関係を格納
        for row in range(sheet2.nrows):
            split = sheet2.cell(row, 1).value.split(':')
            if len(split) == 5 and split[1] == '内評価':
                feature_id_dict.setdefault(split[4],
                                           tuple(map(int, sheet2.cell(row, 0).value.split('-'))))
        feature_id_dict.setdefault('恨み・憎しみ', (1, 2, 1, 2, 1, 2, 0))  # 例外を入力
        feature_id_dict.setdefault('妬み', (1, 2, 1, 2, 1, 2, 0))

        # 評価語の登録--------------------------------------------------------------------------------
        externalized_dict = {}
        polarity2id = {1: 1, 2: -1}
        for row in tqdm(range(1, sheet1.nrows), desc='load Jappraisal'):

            split = sheet1.cell(row, 8).value.split('-')
            polarity, in_ex = polarity2id[int(split[1])], int(split[2])
            if in_ex == 1:
                split = []
                try:
                    split.append(tuple(map(int, sheet1.cell(row, 8).value.split('-'))))
                except ValueError:  # spIDにmultiが含まれる場合
                    for feature in sheet1.cell(row, 9).value.split(':')[4].split('/'):
                        split.append(feature_id_dict[feature])

            for word in sheet1.cell(row, 6).value.split('/'):  # 表記の違いごとに

                word = self.convert_represent_word(word.strip())
                if in_ex == 1:  # 内評価なら
                    self.internalized_dict.setdefault(word, (polarity, split))
                else:  # 外評価なら
                    externalized_dict.setdefault(word, polarity)

                if word in self.impression_dict and self.impression_dict[word] is None:  # 極性がない評価語を補完
                    self.impression_dict[word] = polarity

    def integrate_evaluation_words(self):
        """読み込んだ評価語情報の統合"""

        for word, category in list(self.emotion_dict.items()):
            if word in list(self.internalized_dict.keys()):  # 内評価の評価語内の感情語を削除
                del self.internalized_dict[word]
            if word in list(self.impression_dict.keys()):  # 印象語内の感情語を削除
                del self.impression_dict[word]
            self.emotion_comb_dict.setdefault(word, category)

        for word, info in list(self.internalized_dict.items()):
            if word in list(self.impression_dict.keys()):  # 印象語内の内評価の評価語を削除
                del self.impression_dict[word]

            self.emotion_comb_dict.setdefault(word, info)

            # featureを用いて内評価の評価語に感情カテゴリを付与------------------------------------------
            for feature in info[1]:  # featureごとに
                if feature not in self.internalized2emotion:  # featureと感情カテゴリの関係がない場合は無視
                    continue

                for category in self.internalized2emotion[feature]:  # 関係する感情カテゴリごとに
                    if word in self.emotion_dict:
                        if category in self.emotion_dict[word]:
                            continue
                        self.emotion_dict[word].append(category)
                    else:
                        self.emotion_dict.setdefault(word, [category])
                        del self.internalized_dict[word]
                    self.emotion_category_dict[category].add(word)

                self.emotion_comb_dict[word] = self.emotion_dict[word]  # カテゴリの更新

        # 「だ」の有無の統一------------------------------------------
        for word in list(self.emotion_comb_dict.keys()):
            if word[-1:] == 'だ' and word[:-1] in self.impression_dict:
                del self.impression_dict[word[:-1]]
            elif word + 'だ' in self.impression_dict:
                del self.impression_dict[word + 'だ']
            if word[-1:] != 'だ' and word + 'だ' in self.emotion_comb_dict:
                del self.emotion_comb_dict[word]
        for word in list(self.impression_dict.keys()):
            if word[-1:] != 'だ' and word + 'だ' in self.impression_dict:
                del self.impression_dict[word]


class ParseText(object):
    remove_type = {'副詞', '感動詞', '接続詞', '指示詞', '判定詞', '特殊', '未定義語', '*'}
    stop_words = {'やる', 'する', 'ある', 'いる', 'いえる', 'なる', '思う', '*'}

    def __init__(self, data_dir, adjective_dir, impression_dict, emotion_comb_dict, emotion_dict):

        self.adjective_dir = adjective_dir
        self.impression_dict = impression_dict  # 印象語
        self.emotion_comb_dict = emotion_comb_dict  # 内評価の評価語と感情語
        self.emotion_dict = emotion_dict
        self.dir = data_dir
        self.num_codes = 0
        self.wear_texts = {}
        self.datas = my_functions.load_json('fashion_press.json')

        self.knp_parsed_result = []
        self.types = {'all', 'impression', 'emotion_comb'}
        self.flags = {'negation_word': False, 'negation_conj': False, 'append': False, 'eval': False, 'nai_adn': False}

        self.emotion_categories = ['喜', '怒', '哀', '怖', '恥', '好', '厭', '昂', '安', '驚', '喜(-)', '怒(-)', '哀(-)',
                                   '怖(-)', '恥(-)', '好(-)', '厭(-)', '昂(-)', '安(-)', '驚(-)', '-']
        self.count_impression2emotion = {}
        self.count_emotion_category = {cat: {} for cat in self.emotion_categories}

        self.num_total_words = 0  # 総単語数
        self.count_word = {'all': {}, 'impression': {}, 'emotion_comb': {}, 'adjective': {}, 'noun': {}}  # 単語のカウント
        self.count_link = {'co_occur': {}, 'all': {}, 'direct': {}, 'all_text': {},
                           'direct_text': {}}  # 係り受け関係にある評価語のカウント

        self.adjective_file = 'database_adjective_v20.json'
        if os.path.isfile(self.adjective_dir + '/' + self.adjective_file):
            self.adjective_data = my_functions.load_json(self.adjective_dir + '/' + self.adjective_file)
        else:
            print('形容詞判定ファイルを新規作成')
            self.adjective_data = {}

    def run(self):
        self.knp_parse()
        self.knp_load()

    def knp_parse(self):

        os.chdir(self.dir)

        for k, v in tqdm(self.datas.items(), desc='parse'):
            knp_parsed_result = []
            pool = multiprocessing.Pool(processes=12)
            datas = [data['text'] for data in self.datas[k]]
            for result in tqdm(pool.imap(subprocess_analysis, datas), total=len(datas), desc=f'parse {k}'):
                knp_parsed_result.append(result)

            my_functions.output_json({k: knp_parsed_result}, f'knp_parsed_py_{k}.json')

    def knp_load(self):
        """構文解析結果を読み込み"""

        for genre, genre_data in tqdm(self.datas.items(), desc='parse all'):
            print(genre + ':' + str(len(genre_data)))

            os.chdir(self.dir)
            temp_dic = [[] for _ in range(len(self.datas[genre]))]
            self.texts_lists = {'all': copy.deepcopy(temp_dic),
                                'impression': copy.deepcopy(temp_dic),
                                'emotion_comb': copy.deepcopy(temp_dic),
                                'noun': copy.deepcopy(temp_dic)}
            dic = my_functions.load_json(f'parse_data/knp_parsed_py_{genre}.json')

            print(genre + ':' + str(len(list(dic.values())[0])))

            for num, (knp_text, data) in tqdm(enumerate(zip(list(dic.values())[0], genre_data)), desc=f'parse {genre}'):
                self.text_list = {'all': [], 'impression': [], 'emotion_comb': [], 'noun': []}
                self.page_words = []
                self.parse_text(knp_text)

                for type_ in self.types:  # データを格納
                    self.texts_lists[type_][num] += self.text_list[type_]  # 形態素解析した1つのレビューをリストに格納

            self.output_results(genre)

    def parse_text(self, knp_text):
        """1つのpageの構文解析"""

        for sentence in knp_text:
            links = self.parse_sentence(sentence)

            if not links:
                continue

            self.store_data(links)  # 単語データ格納
            self.count_links(links)

    def parse_sentence(self, sentence):

        self.parsed_sentence = []

        try:
            links = [chunk['parent_id'] for chunk in sentence['bnst']]
        except KeyError:
            return [], []
        for chunk in sentence['bnst']:  # 文節ごとに
            parsed_chunk = self.parse_chunk(chunk['mrph_list'])
            self.parsed_sentence.append(parsed_chunk)
        self.revise_evaluation_type(links)  # 評価語タイプの訂正

        return links

    def parse_chunk(self, chunk):
        """chunkの解析，１つの構文解析結果を出力"""

        # chunk内の単語/品詞を格納
        parsed_chunk = {'word': [], 'word_type': [], 'negation_word_flag': [], 'evaluation_type': [], 'katuyou': [],
                        'midasi': []}
        # 否定，ナイ形容詞語幹，接頭詞,評価語連続のフラグを初期化
        self.flags['negation_word'] = self.flags['negation_conj'] = False
        idxs = []

        for i, word_info in enumerate(chunk):  # chunk内のtokenについて

            if i in idxs:
                continue

            if word_info['katuyou2'] == '語幹' and word_info['hinsi1'] == '形容詞' or word_info['hinsi2'] == 'サ変名詞':
                #  形容詞語幹 or サ変名詞　 + 感　→　印象語に
                flag = self.connect_adjnoun_kan(i, chunk)[0]
                idxs.extend(self.connect_adjnoun_kan(i, chunk)[1])
                new_word = self.connect_adjnoun_kan(i, chunk)[2]
                if flag:
                    parsed_chunk['word'].append(new_word)
                    parsed_chunk['word_type'].append(('名詞', '複合名詞'))
                    parsed_chunk['negation_word_flag'].append(False)
                    if new_word in self.emotion_comb_dict:
                        parsed_chunk['evaluation_type'].append('emotion_comb')
                    else:
                        parsed_chunk['evaluation_type'].append('impression')
                    parsed_chunk['katuyou'].append(('*', '*'))
                    continue

            if word_info['katuyou2'] == '語幹':
                flag = self.connect_adj_noun(i, chunk)[0]
                idxs = self.connect_adj_noun(i, chunk)[1]
                new_word = self.connect_adj_noun(i, chunk)[2]
                if flag:
                    parsed_chunk['word'].append(new_word)
                    parsed_chunk['word_type'].append(('名詞', '複合名詞'))
                    parsed_chunk['negation_word_flag'].append(False)
                    parsed_chunk['evaluation_type'].append(None)
                    parsed_chunk['katuyou'].append(('*', '*'))
                    continue

            if word_info['hinsi1'] == '接頭辞':
                #  接頭辞　+ 名詞　→　連結
                flag = self.connect_settou_noun(i, chunk)[0]
                idxs.extend(self.connect_settou_noun(i, chunk)[1])
                new_word = self.connect_settou_noun(i, chunk)[2]
                if flag:
                    parsed_chunk['word'].append(new_word)
                    parsed_chunk['word_type'].append(('名詞', '複合名詞'))
                    parsed_chunk['negation_word_flag'].append(False)
                    parsed_chunk['evaluation_type'].append(None)
                    parsed_chunk['katuyou'].append(('*', '*'))
                    continue

            if word_info['hinsi1'] == '名詞':
                #  名詞が続いていたら連結
                flag = self.connect_noun(i, chunk)[0]
                idxs.extend(self.connect_noun(i, chunk)[1])
                new_word = self.connect_noun(i, chunk)[2]
                if flag:
                    parsed_chunk['word'].append(new_word)
                    parsed_chunk['word_type'].append(('名詞', '複合名詞'))
                    parsed_chunk['negation_word_flag'].append(False)
                    parsed_chunk['evaluation_type'].append(None)
                    parsed_chunk['katuyou'].append(('*', '*'))
                    continue

            represent_word = word_info['repnames'][0]
            if represent_word:
                self.word = represent_word  # 表記ゆれの統一
            else:
                self.word = word_info['genkei']

            word_type1, word_type2 = word_info['hinsi1'], word_info['hinsi2']  # 品詞情報
            katuyou1, katuyou2 = word_info['katuyou1'], word_info['katuyou2']  # 活用情報

            parsed_chunk['midasi'].append(word_info['midasi'])  # 見出しを追加
            if not self.set_flag(word_type1, word_type2):  # 処理のフラグを立てる
                continue

            negation_word_flag = self.judge_negation_word(word_type1)  # 否定表現かどうか判定
            self.judge_negation_conjunction(word_type1, word_type2, katuyou2)  # 否定表現かどうか判定

            if not self.flags['append']:
                continue

            if not japanese_decision(self.word):
                continue

            if self.flags['eval']:
                evaluation_type = self.judge_evaluation_word(word_type1, word_type2)
                if word_info['genkei'] == 'よい' or word_info['genkei'] == '非常だ' or word_info['genkei'] == 'すごい':
                    if '基本連用形' in katuyou2:
                        evaluation_type = None
            else:
                evaluation_type = None

            parsed_chunk['word'].append(self.word)
            parsed_chunk['word_type'].append((word_type1, word_type2))
            parsed_chunk['negation_word_flag'].append(negation_word_flag)
            parsed_chunk['evaluation_type'].append(evaluation_type)
            parsed_chunk['katuyou'].append((katuyou1, katuyou2))
        if self.flags['negation_word']:  # chunk内が否定の場合
            parsed_chunk['negation_word_flag'] = list(map(operator.not_, parsed_chunk['negation_word_flag']))
        else:  # 否定でない場合（「なくない」なども含まれる）
            parsed_chunk['negation_word_flag'] = [False for _ in parsed_chunk['negation_word_flag']]

        parsed_chunk.setdefault('negation_conj', self.flags['negation_conj'])

        return parsed_chunk

    def connect_adj_noun(self, i, chunk):

        flag = False
        idxs = []
        new_word = ''

        if chunk[i]['hinsi1'] == '形容詞' and chunk[i]['katuyou2'] == '語幹':
            flag = True
            new_word = chunk[i]['midasi']
        if flag == True:
            count = 1
            try:
                while chunk[i + count]['katuyou2'] == '語幹' or chunk[i + count]['hinsi1'] == '名詞':

                    new_word = new_word + chunk[i + count]['midasi']
                    idxs.append(i + count)
                    if chunk[i + count]['hinsi1'] == '名詞':
                        break
                    count += 1
                if chunk[i + 1]['katuyou2'] == '語幹' or chunk[i + 1]['hinsi1'] == '名詞':
                    idxs.append(i)
                else:
                    flag = False
            except:
                flag = False

        return flag, idxs, new_word

    def connect_settou_noun(self, i, chunk):

        flag = False
        new_word = ''
        count = 1
        idxs = []

        if chunk[i]['content_language']:
            new_word = chunk[i]['midasi']
            try:
                if chunk[i + count]['hinsi1'] == '名詞':
                    idxs.append(i)
                    new_word = new_word + chunk[i + count]['midasi']
                    idxs.append(i + count)
                    flag = True
            except:
                flag = False

        return flag, idxs, new_word

    def connect_adjnoun_kan(self, i, chunk):
        #  〜感 → 印象語に（例：重厚感・高級感など）
        flag = False
        idxs = []
        count = 1
        new_word = chunk[i]['midasi']

        try:
            if chunk[i + count]['midasi'] == '感':
                idxs.append(i)
                new_word = new_word + chunk[i + count]['midasi']
                idxs.append(i + count)
                flag = True

        except:
            flag = False

        return flag, idxs, new_word

    def connect_noun(self, i, chunk):

        flag = False
        idxs = []
        new_word = chunk[i]['midasi']

        if chunk[i]['hinsi2'] == '普通名詞' or chunk[i]['hinsi2'] == 'サ変名詞':
            flag = True

        if flag == True:
            count = 1
            try:
                for count in range(1, len(chunk)):
                    if chunk[i + count]['hinsi2'] == '普通名詞' or chunk[i + count]['hinsi2'] == 'サ変名詞':
                        new_word = new_word + chunk[i + count]['midasi']
                        idxs.append(i + count)
                    else:
                        break
                if chunk[i + 1]['hinsi2'] == '普通名詞' or chunk[i + 1]['hinsi2'] == 'サ変名詞':
                    idxs.append(i)
                else:
                    flag = False
            except:
                flag = False

        return flag, idxs, new_word

    def set_flag(self, word_type1, word_type2):
        """処理のフラグを立てる"""

        self.flags['append'] = self.flags['eval'] = False

        # 必要ない品詞，ストップワード，前の単語がナイ形容詞語幹の場合解析しない
        if word_type1 in ParseText.remove_type or self.word in ParseText.stop_words:
            return False
        elif word_type1 in {'動詞', '連体詞', '形容詞'}:
            self.flags['eval'] = self.flags['append'] = True
        elif word_type1 == '名詞':
            self.flags['append'] = True
            if word_type2 in {'普通名詞', 'サ変名詞'}:
                self.flags['eval'] = True
            elif word_type2 == '数詞':
                self.word = 'Number'
            if self.word[-6:] == 'ティ':
                self.word += 'ー'
        elif word_type1 in {'接頭辞', '接尾辞', '助動詞', '助詞'}:
            pass  # 否定表現の判定に用いる
        else:
            return False

        return True

    def judge_negation_word(self, word_type1):
        """否定表現の単語かどうか判定"""

        if (word_type1 == '助動詞' and self.word in {'ない', '無い', 'ぬ', 'まい'}) \
                or (word_type1 == '接尾辞' and self.word == 'ない') \
                or (word_type1 == '接頭辞' and self.word in {'未', '無', '非', '不', '反'}):
            self.flags['negation_word'] = not self.flags['negation_word']
            self.flags['eval'] = False
            return True
        elif (word_type1 == '形容詞' and self.word == '無い') \
                or (word_type1 == '名詞' and self.word == '無し'):
            self.flags['negation_word'] = not self.flags['negation_word']
            self.flags['eval'] = False
            self.flags['nai_adn'] = not self.flags['negation_word']
            return True
        else:
            return False

    def judge_negation_conjunction(self, word_type1, word_type2, katuyou2):
        """否定表現の接続詞かどうか判定"""

        if (word_type1 == '助動詞' and self.word == 'のだ' and katuyou2 == 'ダ列基本連用形') \
                or (word_type1 == '助詞' and word_type2 == '接続助詞' and self.word in {'が', 'けど', 'けれど', 'けども'}):
            self.flags['negation_conj'] = not self.flags['negation_conj']

    def judge_evaluation_word(self, word_type1, word_type2):
        """評価語かどうか判定"""

        evaluation_type = None
        if self.word in self.emotion_comb_dict:
            evaluation_type = 'emotion_comb'
        elif self.word in self.impression_dict:
            evaluation_type = 'impression'
        elif self.word[-1:] == 'だ':  # 「〜だ」なら外して判定
            if self.word[:-1] in self.emotion_comb_dict:
                self.word = self.word[:-1]
                evaluation_type = 'emotion_comb'
            elif self.word[:-1] in self.impression_dict:
                self.word = self.word[:-1]
                evaluation_type = 'impression'
        else:  # 「だ」を加えて判定
            if self.word + 'だ' in self.emotion_comb_dict:
                self.word = self.word + 'だ'
                evaluation_type = 'emotion_comb'
            elif self.word + 'だ' in self.impression_dict:
                self.word = self.word + 'だ'
                evaluation_type = 'impression'

        if word_type1 == '形容詞':
            adjective_word = self.word
            adjective_flag = True
        else:
            adjective_flag, adjective_word = self.judge_adjective(self.word)

        if word_type1 == '名詞' and word_type2 != 'サ変名詞' and not adjective_flag:
            return None  # 評価語ではない
        elif evaluation_type:
            return evaluation_type
        elif adjective_flag:
            self.word = adjective_word
            if self.word in self.emotion_comb_dict:
                return 'emotion_comb'
            else:
                return 'impression'
        else:
            return None  # 評価語ではない

    def judge_adjective(self, word):
        """JUMANの代表表記に変換"""

        if word in self.adjective_data:
            return self.adjective_data[word]
        else:
            result = jumanpp_analysis(word)

            return_set = (False, word)
            if len(result) == 1 and result[0].hinsi == '形容詞':
                represent_word = result[0].repname.split('/')[0].strip()
                if not represent_word:
                    represent_word = result[0].genkei
                return_set = (True, represent_word)
            elif not word[-1:] == 'だ':
                analysis_word = word + 'だ'
                result = jumanpp_analysis(analysis_word)

                if len(result) == 1 and result[0].hinsi == '形容詞':
                    represent_word = result[0].repname.split('/')[0].strip()
                    if not represent_word:
                        represent_word = result[0].genkei
                    return_set = (True, represent_word)

            self.adjective_data.setdefault(word, return_set)
            return return_set

    def store_data(self, links):
        """単語データを格納"""

        self.chunk_evaluation = [{'negation_conj': parsed_chunk['negation_conj'], 'impression': set(),
                                  'emotion_comb': set(), 'various': set()} for parsed_chunk in self.parsed_sentence]
        self.midasis = ''.join([''.join(parsed_chunk['midasi']) for parsed_chunk in self.parsed_sentence])

        for i, (parsed_chunk, link) in enumerate(zip(self.parsed_sentence, links)):

            words = parsed_chunk['word']
            word_types = parsed_chunk['word_type']
            negation_word_flags = parsed_chunk['negation_word_flag']
            evaluation_types = parsed_chunk['evaluation_type']
            katuyou_types = parsed_chunk['katuyou']
            next_parsed_chunk = self.parsed_sentence[link]

            next_words = next_parsed_chunk['word']
            next_word_types = next_parsed_chunk['word_type']

            if link != -1 and \
                    (('無い' in set(next_words) and next_word_types[next_words.index('無い')][0] == '形容詞') or
                     ('無し' in set(next_words) and next_word_types[next_words.index('無し')][0] == '名詞')):
                negation_word_flags = list(map(operator.not_, negation_word_flags))

            for word, word_type, negation_word_flag, evaluation_type, katuyou_type in \
                    zip(words, word_types, negation_word_flags, evaluation_types, katuyou_types):

                if type(word) != str:
                    print(word.decode())
                word += '(-)' if negation_word_flag else ''  # 否定の場合マークを付ける

                if word not in self.page_words:
                    # self.count_one_vocabulary.add(word)
                    # print(self.count_one_vocabulary)
                    self.page_words.append(word)

                    if evaluation_type:
                        self.count_word[evaluation_type][word] = self.count_word[evaluation_type].get(word, 0) + 1

                    elif word_type[0] == '名詞' and word_type[1] not in {'副詞的名詞', '形式名詞', '時相名詞', '数詞'}:
                        self.count_word['noun'][word] = self.count_word['noun'].get(word, 0) + 1

                    self.count_word['all'][word] = self.count_word['all'].get(word, 0) + 1  # 単純頻度のカウント
                self.text_list['all'].append(word)

                if word_type[1] not in {'副詞的名詞', '形式名詞', '時相名詞', '数詞', '組織名'}:
                    if link == -1:
                        continue
                    self.chunk_evaluation[i]['various'].add(word)

                self.num_total_words += 1

                if evaluation_type:
                    self.text_list[evaluation_type].append(word)
                    self.chunk_evaluation[i][evaluation_type].add(word)

    def revise_evaluation_type(self, links):
        """評価語タイプの訂正"""

        for i, (parsed_chunk, link) in enumerate(zip(self.parsed_sentence, links)):  # 文節内の評価語タイプの訂正
            mod_flag = {'flag': False, 'word': '', 'word_type': (), 'evaluation_type': '', 'j': 0}
            for j, (word, evaluation_type, word_type) in \
                    enumerate(zip(parsed_chunk['word'], parsed_chunk['evaluation_type'], parsed_chunk['word_type'])):
                if evaluation_type:
                    if mod_flag['flag']:  # 文節内で評価語が続く場合
                        if mod_flag['evaluation_type'] != evaluation_type:  # 連続する評価語のタイプが異なる場合，印象語をリセット
                            if mod_flag['evaluation_type'] == 'impression':  # 前半が印象語の場合
                                parsed_chunk['evaluation_type'][mod_flag['j']] = None
                                # print "1 [{}] {}".format(mod_flag['word'], parsed_chunk['word'][j])
                            else:  # 後半が印象語の場合
                                parsed_chunk['evaluation_type'][j] = None
                                # print "2 {} [{}]".format(mod_flag['word'], parsed_chunk['word'][j])
                    mod_flag = {'flag': True, 'word': word, 'word_type': word_type, 'evaluation_type': evaluation_type,
                                'j': j}
                else:
                    mod_flag = {'flag': False, 'word': word, 'word_type': None, 'evaluation_type': None, 'j': j}

        for i, (parsed_chunk, link) in enumerate(zip(self.parsed_sentence, links)):  # 文節間の評価語タイプの訂正

            if link == -1:  # リンクの最終の場合ループを抜ける
                break
            elif not parsed_chunk['evaluation_type']:  # 文節内に評価語が含まれない場合スルー
                continue

            if (parsed_chunk['evaluation_type'] and self.parsed_sentence[link]['evaluation_type']) \
                    and (parsed_chunk['evaluation_type'][-1] and self.parsed_sentence[link]['evaluation_type'][0]) \
                    and parsed_chunk['katuyou'][-1][1] == '語幹':
                parsed_chunk['evaluation_type'][-1] = None
                # print "3 [{}] {}".format(parsed_chunk['word'][-1], self.parsed_sentence[link]['word'][0])

    def count_links(self, links):
        """係り受け先のカウント(印象語が出た文節以下で係り受け関係にある文節を全て取得)"""

        for words, link in zip(self.chunk_evaluation, links):
            if not words['impression']:
                continue
            impression_words = words['impression']
            link_flag = {'direct': False, 'all': False}  # 係り受け先に感情後があるかどうかのフラグ
            negation_conj_flag = False  # 否定の接続詞の後かを示すフラグ
            down_link = link
            num_link = 1  # 係り受けの移動数

            if words['negation_conj']:  # 否定接続詞が入っている文節ははじめから反転
                negation_conj_flag = not negation_conj_flag

            while down_link != -1:
                # linked_various_words = self.chunk_evaluation[down_link]['various']
                linked_emotion_words = self.chunk_evaluation[down_link]['emotion_comb']

                for impression_word in impression_words:
                    for emotion_word in linked_emotion_words:

                        if negation_conj_flag:  # 否定表現の接続詞後の場合
                            emotion_word += '*'

                        if down_link == link:  # 直接関係する場合
                            link_flag['direct'] = True
                            self.count_link['direct'][' '.join([impression_word, emotion_word])] = \
                                self.count_link['direct'].get(' '.join([impression_word, emotion_word]), 0) + 1
                        link_flag['all'] = True
                        self.count_link['all'][' '.join([impression_word, emotion_word])] = \
                            self.count_link['all'].get(' '.join([impression_word, emotion_word]), 0) + 1

                if self.chunk_evaluation[down_link]['negation_conj']:  # 係り受け先に否定表現の接続詞があるなら
                    negation_conj_flag = not negation_conj_flag  # 否定の接続詞フラグを反転

                num_link += 1
                down_link = links[down_link]

            if not link_flag['all']:  # 一度も係り先に感情語がなかった場合
                for impression_word in impression_words:
                    self.count_link['all'][' '.join([impression_word, '-'])] = \
                        self.count_link['all'].get(' '.join([impression_word, '-']), 0) + 1

                    if not link_flag['direct']:
                        self.count_link['direct'][' '.join([impression_word, '-'])] = \
                            self.count_link['direct'].get(' '.join([impression_word, '-']), 0) + 1

    def output_results(self, genre):
        """構文解析結果を出力"""

        # jsonファイルにデータを保存

        self.data_dir = self.dir + f'/{genre}'
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        database = my_functions.output_json({}, 'database_review.json', self.data_dir)
        database = my_functions.load_json('database_review.json', self.data_dir)

        for type_ in self.types:
            # 出現頻度の多い順に単語を並べたファイルを出力
            my_functions.output_count_word(self.count_word[type_], 'count_word_{}.tsv'.format(type_), self.data_dir)
            # tsvファイルに形態素解析後のレビューを出力
            database['count_word_{}'.format(type_)] = self.count_word[type_]
            database['texts_list_{}'.format(type_)] = self.texts_lists[type_]

        my_functions.output_count_word(self.count_word['noun'], 'count_word_noun.tsv', self.data_dir)
        my_functions.output_count_word(dict(list(self.count_word['impression'].items()) +
                                            list(self.count_word['emotion_comb'].items())),
                                       'count_word_evaluation.tsv', self.data_dir)
        database['count_word_noun'] = self.count_word['noun']

        for type_, data in self.count_link.items():
            sort_list = output_count_link(self.count_link[type_], 'count_link_{}.tsv'.format(type_), self.data_dir)
            database['count_link_{}'.format(type_)] = sort_list

        for words, count in list(self.count_link['all'].items()):

            impression_word, emotion_word = words.split()

            # 感情語の感情カテゴリの取得
            if emotion_word == '-':
                categories = ['-']
            elif emotion_word in self.emotion_dict:
                categories = self.emotion_dict[emotion_word]
            elif emotion_word[-1:] == '*' and emotion_word[:-1] in self.emotion_dict:  # 否定の感情語の場合
                if emotion_word[:-1] in self.emotion_dict:  # 否定の感情語の場合
                    categories = [cat + '(-)' for cat in self.emotion_dict[emotion_word[:-1]]]
                elif emotion_word[-4:] == '(-)*' and emotion_word[:-4] in self.emotion_dict:  # 否定の感情語の場合
                    categories = self.emotion_dict[emotion_word]
            elif emotion_word[-3:] == '(-)' and emotion_word[:-3] in self.emotion_dict:  # 否定の感情語の場合
                categories = [cat + '(-)' for cat in self.emotion_dict[emotion_word[:-3]]]
            else:  # 感情カテゴリが付与されていない場合
                continue

            if impression_word not in self.count_impression2emotion:
                self.count_impression2emotion.setdefault(impression_word, {cat: 0 for cat in self.emotion_categories})

            for category in categories:
                self.count_impression2emotion[impression_word][category] += count
                self.count_emotion_category[category][emotion_word] = \
                    self.count_emotion_category[category].get(emotion_word, 0) + count

        with open(self.data_dir + '/count_emotion_category.tsv', 'w') as f:
            for category in self.emotion_categories:
                f.write('-- {}\n'.format(category))
                for word, count in sorted(list(self.count_emotion_category[category].items()), key=lambda w: -w[1]):
                    f.write("{}\t{}\t{}\n".format(word, count,
                                                  float(count) / sum(self.count_emotion_category[category].values())))
                f.write("\n")
        database['count_emotion_category'] = self.count_emotion_category
        database['count_impression2emotion'] = self.count_impression2emotion

        information = [['総テキスト数: ' + str(len(self.datas[genre]))],
                       ['総単語数： ' + str(self.num_total_words)],
                       ['語彙数： ' + str(len(self.count_word['all']))],
                       ['評価語数： ' + str(len(self.count_word['impression']) + len(self.count_word['emotion_comb']))],
                       ['印象語数： ' + str(len(self.count_word['impression']))],
                       ['感情語+内評価の語数： ' + str(len(self.count_word['emotion_comb']))]]

        for info in information:
            print((info[0]))

        database['info_parsed'] = information

        my_functions.output_json(database, 'database_review.json', self.data_dir)
        my_functions.output_tsv(information, 'info_parsed.tsv', self.data_dir)


if __name__ == '__main__':
    wear_dir = r'/root/fashion'
    dictionary_dir = r'/root/dictionary'
    represent_dir = r'/root/database'

    load = LoadDictionaryData(data_dir=wear_dir,
                              represent_dir=represent_dir,
                              dictionary_dir=dictionary_dir,
                              load_flag=True)
    dictionary_data = load.run()
    adjective_dir = r'/root/database'

    test = ParseText(data_dir=wear_dir,
                     adjective_dir=adjective_dir,
                     impression_dict=dictionary_data.impression_dict,
                     emotion_comb_dict=dictionary_data.emotion_comb_dict,
                     emotion_dict=dictionary_data.emotion_dict)
    test.run()
