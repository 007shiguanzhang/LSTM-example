# -*- coding: utf-8 -*-
import codecs
import random
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.cross_validation import train_test_split


class DataLoad(object):
    """
    loading data from txt
    """
    def __init__(self):
        self._max_len = 0
        self.x_inputs = []
        self.y_inputs = []
        with codecs.open(u"train.in", 'r', "utf-8") as f:
            x, labes = [], []
            for line in f.readlines():
                line_list = line.split()
                if line_list and line_list[0] != "。":
                    if len(line_list) > 1:
                        x.append(line_list[0])
                        labes.append(line_list[1])
                    else:
                        continue
                else:
                    self.x_inputs.append(x)
                    self.y_inputs.append(labes)
                    if len(x) > self.max_len:
                        self._max_len = len(x)
                    x, labes = [], []
                    # break
        df_data = pd.DataFrame({'words': self.x_inputs, 'tags': self.y_inputs}, index=range(len(self.x_inputs)))
        #  句子长度
        df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
        all_words = list(chain(*df_data['words'].values))
        all_tags = list(chain(*df_data['tags'].values))
        # 2.统计所有 word
        sr_allwords = pd.Series(all_words)
        sr_allwords = sr_allwords.value_counts()
        set_words = sr_allwords.index
        set_ids = range(1, len(set_words)+1)
        tags = pd.Series(all_tags).value_counts()
        set_tags = tags.index
        self.set_tags = set_tags
        tag_ids = range(len(set_tags))
        # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
        self.word2id = pd.Series(set_ids, index=set_words)
        self.id2word = pd.Series(set_words, index=set_ids)
        self.tag2id = pd.Series(tag_ids, index=set_tags)
        self.id2tag = pd.Series(set_tags, index=tag_ids)
        max_len = self._max_len
        self.words = len(set_words), len(set_tags)
        def X_padding(words):
            """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
            ids = list(self.word2id[words])
            if len(ids) >= max_len:  # 长则弃掉
                return ids[:max_len]
            ids.extend([0] * (max_len - len(ids)))  # 短则补全
            return ids

        def y_padding(tags):
            """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
            ids = list(self.tag2id[tags])
            if len(ids) >= max_len:  # 长则弃掉
                return ids[:max_len]
            ids.extend([0] * (max_len - len(ids)))
            return ids
        df_data['X'] = df_data['words'].apply(X_padding)
        df_data['y'] = df_data['tags'].apply(y_padding)
        self._X = np.asarray(list(df_data['X'].values))
        self._y = np.asarray(list(df_data['y'].values))
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._X, self._y, test_size=0.2,
                                                                                random_state=42)

    @property
    def length(self):
        return self.words

    @property
    def y(self):
        return self._y

    @property
    def max_len(self):
        return self._max_len

    @property
    def train(self):
        return self.X_train, self.y_train

    @property
    def test(self):
        return self.X_test, self.y_test

    def test2ids(self, text):
        words = list(text)
        ids = list(self.word2id[words])
        if len(ids) >= self._max_len:  # 长则弃掉
            print(u'输出片段超过%d部分无法处理' % (self._max_len))
            return ids[:self._max_len]
        ids.extend([0] * (self._max_len - len(ids)))  # 短则补全
        ids = np.asarray(ids).reshape([-1, self._max_len])
        return ids


def split_train(inputs):
    result = []
    for ii in inputs:
        k = 0
        while k < len(ii):
            result.append(ii[k:k+32])
            k += 16
        result.pop()
    return result


def transform(inputs, class_nums):
    result = []
    for index, item in enumerate(inputs):
        temp = [0]*class_nums
        temp[item] = 1
        result.append(temp)
    return result
if __name__ == '__main__':
    # data = DataLoad()
    # print(data.x_inputs, '\n', data.y_inputs)
    # result = split_train(data.X_test)
    class_nums = 19
    inputs = [random.randint(0, 18) for i in range(10)]
    result = transform(inputs, class_nums)
    print('inputs:', inputs)
    for index, item in enumerate(result):
        print('result:', inputs[index], item)
