import codecs
import pandas as pd
import numpy as np
import random
from itertools import chain
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
np.set_printoptions(threshold=np.inf) 
class DataLoad(object):
    """
    loading data from txt
    """
    def __init__(self):
        self._max_len = 0
        i=0
        self.time_steps=time_steps=24
        self.x_inputs = []
        self.y_inputs = []
        #with codecs.open(u"/home/xjj/lstm2med/train.in", 'r', "utf-8") as f:
        with codecs.open(u"train.in", 'r', "utf-8") as f:
            x, labes = [], []
            for line in f.readlines():
                #print(line)
                line_list = line.split()
                #print(line_list)
                if line_list != []:
                    #print(line_list)
                    if len(line_list) == 2:
                        x.append(line_list[0])
                        labes.append(line_list[1])
                    else:
                        x.append(" ")
                        labes.append(line_list[0])
 #                   x.append(line_list[0])
#                    if len(line_list) == 2 :
#                        labes.append(line_list[1])
        
                else:
                    self.x_inputs.append(x)
                    i+=1
                    self.y_inputs.append(labes)
                    if len(x) > self.max_len:
                        self._max_len = len(x)
                        if (self._max_len>1100):
                            print(self._max_len,i)
                           # print(x)
                    x, labes = [], []
                    #break
        df_data = pd.DataFrame({'words': self.x_inputs, 'tags': self.y_inputs}, index=range(len(self.x_inputs)))
        #print(df_data)
        #  句子长度
        df_data['sentence_len'] = df_data['words'].apply(lambda words: len(words))
        all_words = list(chain(*df_data['words'].values))
        #print(all_words)
        all_tags = list(chain(*df_data['tags'].values))
        # 2.统计所有 word
        sr_allwords = pd.Series(all_words)
        #print(sr_allwords)
        sr_allwords = sr_allwords.value_counts()
        #print(sr_allwords)
        self.set_words = sr_allwords.index
        #print(self.set_words)
        set_ids = range(1, len(self.set_words)+1)
        #print(set_ids)
        tags = pd.Series(all_tags).value_counts()
        self.set_tags = tags.index
        tag_ids = range(len(self.set_tags))
        # 3. 构建 words 和 tags 都转为数值 id 的映射（使用 Series 比 dict 更加方便）
        self.word2id = pd.Series(set_ids, index=self.set_words)
        self.id2word = pd.Series(self.set_words, index=set_ids)
        self.tag2id = pd.Series(tag_ids, index=self.set_tags)
        self.id2tag = pd.Series(self.set_tags, index=tag_ids)
        word2id = pd.Series(set_ids, index=self.set_words)
        max_len = self._max_len
        self.words = len(self.set_words), len(self.set_tags)
        print(self.words)
        
        def X_padding(words):
            """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
            ids = list(self.word2id[words])
            if len(ids) <= time_steps:
                ids.extend([0]*(time_steps - len(ids)))
            else:
                ids.extend([0]*int((time_steps/3)-len(ids)%(time_steps/3)))
            #if len(ids) >= max_len:  # 长则弃掉
#                return ids[:max_len]
#            ids.extend([0] * (max_len - len(ids)))  # 短则补全
            return ids

        def y_padding(tags):
            """把 tags 转为 id 形式， 并自动补全位 max_len 长度。"""
            ids = list(self.tag2id[tags])
            if len(ids) <= time_steps:
                ids.extend([0]*(time_steps - len(ids)))
            else:
                ids.extend([0]*int((time_steps/3)-len(ids)%(time_steps/3)))
            #if len(ids) >= max_len:  # 长则弃掉
 #               return ids[:max_len]
 #           ids.extend([0] * (max_len - len(ids)))
            return ids
        df_data['X'] = df_data['words'].apply(X_padding)
        #print(df_data['X'])
        df_data['y'] = df_data['tags'].apply(y_padding)
       # for i in range (len(df_data['X'].values)):
#            for j in range (len(df_data['X'].values[i])):
#                df_data['X'].values[i][j]=df_data['X'].values[i][j]/len(self.set_words)
        self._x = np.asarray(list(df_data['X'].values))
        #print(len(self.set_words))
        #print(df_data['X'].values[0])
        #print(len(df_data['X'].values[0]))
        #print(self._x)
        self._y = np.asarray(list(df_data['y'].values))
        # 划分训练集和测试集
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self._x, self._y, test_size=0.2,random_state=42)

        #print(len(self.x_test[0]))
        self.x_train2lstm = self.split_train(self.x_train)
        self.y_train2lstm = self.split_train(self.y_train)
 #       print(len(self.x_train2lstm))
        #print(self.x_train2lstm[0:10])
        self.x_train2lstm,self.y_train2lstm = self.random_sort(self.x_train2lstm,self.y_train2lstm)
        self.x_train2lstm = np.asarray(self.x_train2lstm)
        self.y_train2lstm = np.asarray(self.y_train2lstm)
        #print(self.x_train2lstm)
       # print(self.y_train2lstm)
        #print(self.x_train2lstm[0:100])
        self.x_test2lstm = []
        self.y_test2lstm = []
        for i in range(len(self.x_test)):
            testdata = self.split_test(self.x_test[i])
            self.x_test2lstm.append(testdata)
        for i in range(len(self.y_test)):
            testlabel = self.split_test(self.y_test[i])
            self.y_test2lstm.append(testlabel)
        #print(self.x_test2lstm[0])
        #print(self.y_test2lstm)
        #print(self.y_test2lstm[0:1])
        #self.x_test2lstm = np.asarray(self.split_train(self.x_test))
        #self.y_test2lstm = np.asarray(self.split_train(self.y_test))
 #       aa=self.embedding_y(self.y_train)
        #print(self.x_train[2])
        #print(len(self.x_train[2]))
        #print(aa)
        #print(len(self.set_words))
        #print(self.x_train2lstm[0])
        #print(self.x_train2lstm,self.x_test2lstm)

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
        return self.x_train2lstm, self.y_train2lstm

    @property
    def test(self):
        return self.x_test2lstm, self.y_test2lstm
    
    def random_sort(self, inputs, labels):
        temp = list(range(len(inputs)))
        random.shuffle(temp)
        result_inputs, reslut_labels = [], []
        for index in temp:
            result_inputs.append(inputs[index])
            reslut_labels.append(labels[index])
        return result_inputs, reslut_labels

    def split_train(self,inputs):
        result =[]
        for ii in inputs:
            k = 0
            while k < len(ii)-self.time_steps/3:
                result.append(ii[k:k+self.time_steps])
                k +=int(self.time_steps/3)
            result.pop()
        return result
    def split_test(self,inputs):
        result = []
        k = 0
        while k<len(inputs)-self.time_steps/3:
            result.append(inputs[k:k+self.time_steps])
            k += int(self.time_steps/3)
        result.pop()
        return result
            
    def embedding_y(self,label):
        result = []
        z = np.zeros(len(self.set_tags))
        for i in label:
            for j in i:
                z1 = z
                z1[j]=1
                result.append(z1)
        result.pop()
        return result
            
    def test2ids(self, text):
        words = list(text)
        ids = list(self.word2id[words])
        if len(ids) >= self._max_len:  # 长则弃掉
            print(u'输出片段超过%d部分无法处理' % (self._max_len))
            return ids[:self._max_len]
        ids.extend([0] * (self._max_len - len(ids)))  # 短则补全
        ids = np.asarray(ids).reshape([-1, self._max_len])
        return ids
#a=DataLoad()
