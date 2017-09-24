#!/usr/bin/python
# coding:utf-8
from data import DataLoad
from Batch import BatchGenerator
from model import Model


def max_index(s):
        index_max = 0
        for ii in range(len(s)):
            if s[ii] > index_max:
                index_max = ii
        return index_max
if __name__ == '__main__':
    data = DataLoad()
    length_words, length_tags = data.words
    max_len = data.max_len
    x_train, y_train = data.train
    x_test, y_test = data.test
    id2tag = data.id2tag
    id2word = data.id2word
    print('Creating the data generator ...')
    data_train = BatchGenerator(x_train, y_train, shuffle=True)
    data_test = BatchGenerator(x_test, y_test, shuffle=False)
    print('Finished creating the data generator.')
    # building model
    test = u' '
    ids = data.test2ids(test)
    model = Model(max_len, length_words, length_tags, data_train, data_test, id2word, id2tag)
    # out_ids = model.out_test(test, ids)
    # print(out_ids)
    result = []
    # for i in model._y_pred:
    #     result.append(max_index(i))
    # print(model._y_pred)
    # tags = id2tag[result]
    # print(test, tags.values)
