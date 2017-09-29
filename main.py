from data import DataLoad
from Batch import BatchGenerator
from model3 import Model
import tensorflow as tf
import numpy as np

np.set_printoptions(threshold=np.inf) 
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
    id2tag, id2word = data.id2tag, data.id2word
    id2 = (id2word, id2tag)
    # 构建输入数据类
    print('Creating the data generator ...')
    data_train = BatchGenerator(x_train, y_train, shuffle=True)
    print(data_train.num_examples)
    a, b = data_train.next_batch(10)
    data_test = BatchGenerator(x_test, y_test, shuffle=False)
    print('Finished creating the data generator.')

    # building model
    test = u'中年男性，48岁，主因：腹痛、腹胀3天。'
    ids = data.test2ids(test)
    model = Model(id2, length_words, length_tags, data_train, data_test)
    result = []
