import requests
import lxml.html
import json
import re
from tqdm import tqdm

try:
    from my_packages import my_functions
except ImportError:
    import my_functions

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

class ScrapingWear(object):
    def __init__(self):
        self.rank_page_url = 'https://www.fashion-press.net/news/search/'
        self.ganres =  ['fashion','beauty','gourmet','art','movie','music','lifestyle']
        self.data = {'fashion':[],'beauty':[],'gourmet':[],'art':[],'movie':[],'music':[],'lifestyle':[]}

    def run(self):
        for genre in self.ganres:
            print(f'{genre} start')
            self.rank_page(genre)

    def rank_page(self, genre):
        page = 1
        page_num = 0
        timeline = 2019
        with tqdm() as pbar:

            while timeline > 2010:
                rank_html = requests.get(self.rank_page_url + genre, params = {'page' :page}).text
                rank_root = lxml.html.fromstring(rank_html)
                for i in range(1,30):
                    page_num += 1
                    page_data = {}

                    try:
                        pbar.update(1)
                        url = str(rank_root.cssselect(f'body > div.fp_wrapper > div.pc_only > div > div:nth-child({i+1}) > a')[0].get('href'))
                        html = requests.get('https://www.fashion-press.net' + url)
                        root = lxml.html.fromstring(html.text)
                        title = str(root.cssselect(
                            'body > div.fp_wrapper > div > div.fp_content_left > div.h1_wrapper > h1')[
                                        0].text_content())
                        time = str(root.cssselect(
                            'body > div.fp_wrapper > div > div.fp_content_left > div.footer_caption > span:nth-child(1)')[
                                       0].text_content())

                        page_data['title'] = title
                        page_data['time'] = time.replace('更新', '')
                        timeline = int(page_data['time'][:4])
                        page_data['url'] = 'https://www.fashion-press.net' + url
                        page_data['text'] = ''
                        html.encoding = html.apparent_encoding
                        sentences = re.split('<!-- default -->|関連ニュース|【問い合わせ先】|【アイテム詳細】|【詳細】|【特集】|【映画】|<!-- pager -->',
                                             html.text)  # 文ごとにレビューをカット
                        test = [word for word in re.split('<|>', sentences[1]) if
                                '/' not in word and re.search(r'[ぁ-んァ-ン]', word)]
                        page_data['text'] += ('').join(test)

                        if '<ul class="pagination-list">' in html.text:
                            for j in range(2, 4):
                                html = requests.get('https://www.fashion-press.net' + url + f'/{j}')
                                html.encoding = html.apparent_encoding
                                if 'Not Found' in html.text:
                                    continue

                                sentences = re.split('<!-- default -->|関連ニュース|【問い合わせ先】|【アイテム詳細】|【詳細】|【特集】|【映画】|<!-- pager -->',html.text)  # 文ごとにレビューをカット
                                test = [word for word in re.split('<|>',sentences[1]) if '/' not in word and re.search(r'[ぁ-んァ-ン]',word)]
                                page_data['text'] += ('').join(test)

                        if page_data not in self.data[genre]:
                            self.data[genre].append(page_data)
                        my_functions.output_json(self.data, 'outputs/fashion_press.json')

                    except:
                        continue

                page += 1

if __name__ == '__main__':

    test = ScrapingWear()
    test.run()
