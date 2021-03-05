import pandas as pd
import numpy as np

try:
    from my_packages import my_functions
except ImportError:
    import my_functions

def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x - xmean) / xstd
    return zscore

class TimeAnalysis(object):

    def __init__(self):
        self.all_original_data = my_functions.load_json('outputs/fashion_press.json')
        self.count_impression = {}

        self.timeline_data = {}
        self.count_data = {}
        self.parse_num = {}
        self.frequency_data = {}

        self.spring = ['03', '04', '05']
        self.summer = ['06', '07', '08']
        self.autumn = ['09', '10' ,'11']
        self.winter = ['12', '01', '02']

        self.impression_words = []
        self.database = {}

    def run(self):
        self.count_impression_word()
        self.modification_to_timeline_data()
        self.count_per_timeline()
        self.compare_fashion_word()
        my_functions.output_json(self.count_data, 'outputs/count_data.json')

    def count_impression_word(self):
        """ 印象語の選定"""
        for genre, original_data in self.all_original_data.items():
            parse_data = my_functions.load_json(f'outputs/{genre}/database_newsarticles.json')
            word_count = {}
            text_num = int(parse_data['info_parsed'][0][0].strip('総テキスト数: '))
            for text in parse_data['texts_list_impression']:
                for word in text:
                    word_count[word] = word_count.get(word, 0) + 1

            for word, count in word_count.items():
                word_count[word] = count / text_num
            self.count_impression[genre] = word_count

        impression_words =[w for w,c in self.count_impression['fashion'].items() if c > 0.005]
        self.impression_words = []
        for word in impression_words:
            other_counts = 0
            fashion_count = self.count_impression['fashion'][word]
            other_len = 0
            for genre in ['beauty', 'gourmet', 'art', 'movie', 'music', 'lifestyle']:
                try:
                    other_len += 1
                    other_counts += self.count_impression[genre][word]
                except:
                    continue
            other_count = other_counts / other_len
            if (fashion_count - other_count) > 0.005:
                self.impression_words.append(word)

    def modification_to_timeline_data(self):
        """ 季節ごとにデータを分割"""
        for genre, original_data in self.all_original_data.items():
            timeline_data = {str(i):{'spring':[], 'summer':[], 'autumn':[], 'winter':[]} for i in range(2010,2021)}
            self.parse_data = my_functions.load_json(f'outputs/{genre}/database_newsarticles.json')

            for text, genre_data in zip(self.parse_data['texts_list_impression'], original_data):
                if genre_data['time'][:7].replace('-','')[-2:] in self.spring:
                    timeline_data[genre_data['time'][:4]]['spring'].append(text)
                if genre_data['time'][:7].replace('-','')[-2:] in self.summer:
                    timeline_data[genre_data['time'][:4]]['summer'].append(text)
                if genre_data['time'][:7].replace('-','')[-2:] in self.autumn:
                    timeline_data[genre_data['time'][:4]]['autumn'].append(text)
                if genre_data['time'][:7].replace('-','')[-2:] in self.winter:
                    timeline_data[genre_data['time'][:4]]['winter'].append(text)
            self.timeline_data[genre] = timeline_data

    def count_per_timeline(self):
        for genre, timeline_data in self.timeline_data.items():
            self.count_data[genre] = {str(i):{'spring':{}, 'summer':{}, 'autumn':{}, 'winter':{}}  for i in range(2010,2021)}
            self.parse_num[genre] = {str(i):{'spring':'', 'summer':'', 'autumn':'', 'winter':''}  for i in range(2010,2021)}
            self.frequency_data = {str(i):{'spring':{}, 'summer':{}, 'autumn':{}, 'winter':{}}  for i in range(2010,2021)}

            for time, dic in timeline_data.items():
                for season, texts in dic.items():
                    self.parse_num[genre][time][season] = len(texts)
                    print(genre, time, season, len(texts))
                    for text in texts:
                        count_data = []
                        for word in text:
                            if word in self.impression_words:
                                self.count_data[genre][time][season][word] = self.count_data[genre][time][season].get(
                                    word, 0) + 1
                                count_data.append(word)
                    for k, v in self.count_data[genre][time][season].items():
                        self.frequency_data[genre][time][season][k] = v / len(texts)


    def compare_fashion_word(self):
        df = pd.DataFrame()

        for time, dic in self.frequency_data['fashion'].items():
            for season, s_dic in dic.items():
                words = list(s_dic.keys())
                frequency = list(s_dic.values())
                s = pd.Series(frequency, index=words, name=f'{time}_{season}')
                df = df.append(s)

        self.df = (df.fillna(0)).T
        self.impression_words = self.df.index.values
        self.fashion_times = self.df.columns.values
        self.season_vecs = zscore(self.df.values, axis=1).tolist()

    def output(self):

        self.database['impression_words']  = self.impression_words
        self.database['season_vecs'] = self.season_vecs
        self.database['fashion_times'] = self.fashion_times

        my_functions.output_json(self.database, 'outputs/database_vecs.json')

if __name__ == '__main__':
    test = TimeAnalysis()
    test.run()