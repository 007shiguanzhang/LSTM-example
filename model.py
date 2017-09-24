# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, max_len, vocab_size, class_num, data_train, data_test, id2word, id2tag):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # config
        self.decay = 0.85
        self.max_epoch = 5
        self.max_max_epoch = 2 #10
        self.timestep_size = max_len  # 句子长度
        self.vocab_size = vocab_size  # 样本中不同字的个数，根据处理数据的时候得到
        self.class_num = class_num  # class_num 分类数目
        self.input_size = self.embedding_size = 128  # 字向量长度
        self.hidden_size = 128  # 隐含层节点数
        self.layer_num = 5  # lstm 层数
        self.max_grad_norm = 5.0  # 最大梯度（超过此值的梯度将被裁剪）

        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

        with tf.variable_scope("Inputs"):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')

        bilstm_out = self.bi_lstm(self.X_inputs)

        with tf.variable_scope('outputs'):
            softmax_w = self.weight_variable([self.hidden_size*2, self.class_num])
            softmax_b = self.bias_variable([self.class_num])
            self.y_pred = tf.matmul(bilstm_out, softmax_w) + softmax_b
        self.outputs = tf.reshape(self.y_pred, [self.batch_size, -1, self.class_num])
        # y_inputs.shape = [batch_size, timestep_size]
        self.seq_length = tf.convert_to_tensor(self.batch_size * [max_len], dtype=tf.int32)
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y_inputs, self.seq_length)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        print('Finished creating the lstm model.')

        print('start training')
        self.sess.run(tf.global_variables_initializer())
        _batch_size = 50
        batch_num = int(data_train.y.shape[0] / _batch_size)
        fetches = [self.outputs, self.transition_params, self.y_pred]
        for i in range(batch_num):
            x_batch, y_batch = data_train.next_batch(_batch_size)
            feed_dict = {self.X_inputs: x_batch, self.y_inputs: y_batch, self.lr: 1e-5, self.batch_size: _batch_size,
                         self.keep_prob: 1.0}
            scores, transition_params, y_pred = self.sess.run(fetches, feed_dict)
            # if i % 100 == 0:
            #     print("scores:", scores, '\n', "y_pred:", y_pred)
        print("start test")
        _batch_size = 50
        fetches = [self.outputs, self.transition_params]
        _y = data_test.y
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size)
        for i in range(batch_num):
            x_batch, y_batch = data_test.next_batch(_batch_size)
            feed_dict = {self.X_inputs: x_batch, self.y_inputs: y_batch, self.lr: 1e-5, self.batch_size: _batch_size,
                         self.keep_prob: 1.0}
            scores, transition_params = self.sess.run(fetches, feed_dict)
            tf_unary_scores = np.squeeze(scores)
            _accs = 0.0
            words, tags, viterbi_sequence = [], [], []
            for index, ii in enumerate(tf_unary_scores):
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    ii, transition_params)
                words = list(id2word[x_batch[index]])
                tags = list(id2tag[y_batch[index]])
                acc = self.acc(viterbi_sequence, y_batch[index])
                _accs += acc
            if i % 30 == 0:
                print("words:", words, "tags:", tags, "result:", viterbi_sequence)
                print('test acc:', _accs/len(tf_unary_scores))

    def bi_lstm(self, X_inputs):
        # building
        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
        inputs = tf.nn.embedding_lookup(embedding, X_inputs)
        cell_fw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
        # ** 4.初始状态
        initial_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
        # ***********************************************************
        # ** 5. bi-lstm 计算（展开）
        with tf.variable_scope('bidirectional_rnn'):
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw'):
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw'):
                inputs = tf.reverse(inputs, [1])
                for timestep in range(self.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            output = tf.concat([outputs_fw, outputs_bw], 2)
            # output = tf.transpose(output, perm=[1, 0, 2])
            output = tf.reshape(output, [-1, self.hidden_size * 2])
        return output

    @staticmethod
    def lstm_cell():
        cell = rnn.LSTMCell(128, tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=1.0)

    @staticmethod
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def out_test(self, test, ids):
        if test:
            text_len = len(test)  # 这里每个 batch 是一个样本
            X_batch = ids
            fetches = [self.y_pred]
            feed_dict = {self.X_inputs: X_batch, self.lr: 1.0, self.batch_size: 1, self.keep_prob: 1.0}
            _y_pred = self.sess.run(fetches, feed_dict)[0][:text_len]  # padding填充的部分直接丢弃
            return _y_pred
        else:
            return []

    @staticmethod
    def acc(pred, label):
        acc = 0
        for index, item in enumerate(pred):
            if item == label[index]:
                acc += 1
        return acc/len(pred)
