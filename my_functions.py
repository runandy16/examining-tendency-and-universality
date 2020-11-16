# coding=utf-8

from __future__ import unicode_literals
import os
import re
import csv
import sys
import json


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def json_load_byteified(file_handle):
    return _byteify(json.load(file_handle, object_hook=_byteify), ignore_dicts=True)


def _byteify(data, ignore_dicts=False):
    """utf-8にjsonを変換"""

    if isinstance(data, str):
        return data.encode('utf-8')
    elif isinstance(data, list):
        return [_byteify(item, ignore_dicts=True) for item in data]
    elif isinstance(data, dict) and not ignore_dicts:
        return {_byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True) for key, value in data.items()}
    else:
        return data


def load_json(load_file, load_dir=''):
    if load_dir:
        load_file = load_dir + '/' + load_file

    with open(load_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    return loaded_data

def output_tsv(data, output_file, output_dir='', type_='w'):
    """データをTSVファイルに出力"""

    if output_dir:
        output_file = output_dir + '/' + output_file

    with open(output_file, type_) as f_write:
        writer = csv.writer(f_write, delimiter='\t', quotechar=None)
        writer.writerows(data)


def output_count_word(dictionary, output_file, directory):
    """カウントした辞書を出力"""

    sort_list = sorted(list(dictionary.items()), key=lambda x: x[1])
    sort_list.reverse()
    output_tsv(sort_list, output_file, directory)


def output_json(data, output_file, output_dir='', type_='w'):
    """データをTSVファイルに出力"""

    if output_dir:
        output_file = output_dir + '/' + output_file

    with open(output_file, type_,encoding='utf-8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False, cls=SetEncoder))
        # f.write(json.dumps(data, indent=4, ensure_ascii=False))


def output_json_en(data, output_file, output_dir='', type_='w'):
    """データをTSVファイルに出力"""

    if output_dir:
        output_file = output_dir + '/' + output_file

    with open(output_file, type_) as f:
        f.write(json.dumps(data, indent=4))


def line_notification(message='Complete'):
    """LINEで通知"""

    import requests

    line_notify_token = 'LcT5z6kVbkOiRCCVINb7wL7WeLhX24omXCw5EYxVY0p'
    # ↑やまだるなのアクセストークン
    line_notify_api = 'https://notify-api.line.me/api/notify'

    requests.post(line_notify_api,
                  data={'message': message},
                  headers={'Authorization': 'Bearer ' + line_notify_token})


def trimming_images(x, y, w, h, input_dir, output_dir):
    """画像を指定位置でトリミング"""

    import cv2

    os.chdir(input_dir)
    files = os.listdir(input_dir)

    for file_ in files:
        img = cv2.imread(file_)
        print((img.shape))

        cv2.imshow('image', img)

        dst = img[y:y + h, x:x + w]

        os.chdir(output_dir)
        cv2.imwrite(file_, dst)
        os.chdir(input_dir)


def conv_encoding(input_dir, input_file):
    """ファイル読込とエンコーディングの調査"""

    file_ = open(input_dir + input_file, 'r')

    try:
        data = file_.read()

        lookup = ('utf_8', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213',
                  'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
                  'iso2022jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_3',
                  'iso2022_jp_ext', 'latin_1', 'ascii')
        encode = None
        for encoding in lookup:
            try:
                data = data.decode(encoding)
                encode = encoding
                break
            except Exception as e:
                print(e)
                sys.exit()
        if isinstance(data, str):
            print(encode)
            return data, encode
        else:
            raise LookupError
    finally:
        file_.close()


def convert_str(words):
    """str型に変換"""

    try:
        return str(words)
    except UnicodeEncodeError:  # "’"など変な記号を除去
        temp = ''
        for w in words:  # 各文字ごとにエラーが出る文字を除去
            try:
                temp += str(w)
            except UnicodeEncodeError:
                pass

    return temp


def flatten(nested_list):
    """2重のリストをフラットにする関数"""
    return [e for inner_list in nested_list for e in inner_list]


def load_tsv(file):
    csv.field_size_limit(1000000000)
    with open(file) as f:
        reader = csv.reader(f)
        file_list = [row[0].split('\t') for row in reader]
    return file_list

def japanese_decision(word):
    if re.match('[ぁ-んァ-ン一-龥]',word):
        return True
    else:
        return False

