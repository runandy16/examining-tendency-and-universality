# examining-tendency-and-universality

### Buillding examining-tendency-and-universality
examining-tendency-and-universalityは印象の普遍性と時代性の切り分けを行うデモツールです．
次のコマンドでDockerによる環境構築を行ってください．

``` 
docker build -t knp_neologd:python3 /examining_tendency_and_universality/knp_neologd:python3
``` 

### データ収集

本実験はFashion Pressのニュース記事を使用します．

``` scraping_fashionpress.py``` では，2010年以降の全カテゴリのニュース記事を収集します．

- 'fashion','beauty','gourmet','art','movie','music','lifestyle' の全7カテゴリ
- ``` fashion_press.json``` の中身
``` 
{
    "fashion": [
        {
            "title": "記事タイトル",
            "time": "2020-11-16 10:00  ",
            "url": "https://www.fashion-press.net/news/id",
            "text": "記事"
        }　…    ], 
        
    "beauty" : [
        {
            "title": "記事タイトル",
            "time": "2020-11-16 12:00  ",
            "url": "https://www.fashion-press.net/news/id",
            "text": "記事"
        }　…    ] , … 
}
``` 

### 前処理

収集したニュース記事のテキストを前処理（形態素解析+係り受け解析）します。

```
python parse_newsarticles.py 
```

- ``` knp_result``` : 各カテゴリごとの形態素解析の結果  
- ``` count_emotion_category.tsv``` : 感情語の出現回数  
- ``` count_link_direct.tsv``` : 印象語の感情語との直接の係り受けの生起回数         
- ``` count_word_evaluation.tsv``` : 評価語の出現回数  
- ``` count_link_all.tsv``` : 印象語と感情語との直接的な係り受けと間接的な係り受けの生起回数を合わせたもの
- ``` count_word_all.tsv``` : 単語の出現頻度
- ``` count_word_impression.tsv``` : 印象語の出現頻度
- ``` info_parsed.tsv``` : 形態素解析結果の概要。中身は以下
    - 総テキスト数: 総テキスト数
    - 総単語数: 単語の出現回数
    - 語彙数: 単語の種類
    - 評価語数: 評価語の出現回数
    - 印象語数: 印象語の出現回数
    - 感情語(+内評価)の語数
- ``` count_link_co_occur.tsv``` : 印象語と感情語の共起回数
- ``` count_word_emotion_comb.tsv``` : 製品ごとに見たときに感情語が出てきた回数
- ``` count_word_noun.tsv``` : 名詞をの出現頻度
- ``` {カテゴリ}/database_review.json``` : すべてのデータベースを格納


### 印象語の時系列データ化

他カテゴリとの印象語出現頻度の比較をします。
ファッションカテゴリにおいて以下の条件を満たす印象語を抽出し，期間ごとに出現確率を求めて標準化を行います。
- 0.5% 以上のページに出現
- 他ドメインカテゴリでの出現確率との差分が0.5 % 以上

```
python pop_frequency_comparison.py
```

- ``` outputs/database_vecs.json```: 選定後の印象後のリスト/時系列ごとの印象語の出現確率

### 状態空間モデルによる時系列分析

作成した時系列データをロードし、状態空間モデルを構築します。
モデル構築後、潜在的な時系列傾向である内部状態または季節変動をクラスタリングします（AICによってクラスタ数を定める）．

```
python clustering_fluctuation.py
```
- ```impressions_trend/{word}_trend.png```: {印象後}の状態空間モデルによる要因分解結果
- ```impressions_trend```: 要因分解結果をk-meansによりクラスタリングした出力ファイル


### Word2Vecによる感性評価指標の抽出

時系列傾向のパターンごとにWord2Vecを用いた階層クラスタリングを行う。

```
python impression_clustering.py
```
