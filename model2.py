import tensorflow as tf
import numpy as np
tf.set_random_seed(1)   # set random seed

# 导入数据
np.set_printoptions(threshold=np.inf) 
class Model(object):
    def __init__(self, max_len, vocab_size, class_num, data_train, data_test, test, ids):

        #  hyperparameters
        self.lr = 0.001                  # learning rate
        self.training_iters = 9011     # train step 上限
        self.batch_size = 10
        self.num_layers = 2
        self.vocab_size = vocab_size   #样本中不同字的个数，处理数据的时候得到
        self.n_inputs = self.embedding_size = 128  # MNIST data input (img shape: 28*28)
        self.n_steps = 16                # time steps
        self.n_hidden_units = 128        # neurons in hidden layer
        self.n_classes = class_num              # MNIST classes (0-9 digits)

# x y placeholder
        x = tf.placeholder(tf.int32, [None, self.n_steps])
        y = tf.placeholder(tf.float32, [None,self.n_classes])

# 对 weights biases 初始值的定义
        weights = {
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }
        pred = self.RNN(x, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        print('Finished creating the lstm model.')
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            step = 0
            acc_list = []
            while step * self.batch_size < self.training_iters:
                batch_xs, batch_ys = data_train.next_batch(self.batch_size)
                batch_y = np.asarray(self.embedding_y(batch_ys))
                sess.run([train_op], feed_dict={
                    x: batch_xs,
                    y: batch_y,
                })
                if step % 100 == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_y,})
                    acc_list.append(acc)
                    print(acc)
                    print(sess.run(tf.argmax(y, -1), feed_dict={
                    x: batch_xs,
                    y: batch_y,
                }))
#                    print(sess.run(pred[0],feed_dict={
#                    x: batch_xs,
#                    y: batch_y,
#                }))
                step += 1
            print('acc变化:', acc_list)
    def RNN(self, X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (10 batches , 16 steps, 64 inputs)
        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
        X = tf.nn.embedding_lookup(embedding, X)
    # X ==> (10 batches * 16 steps, 64 inputs)
        # X = tf.cast(X, tf.float32)
        # X = tf.reshape(X, [-1, self.n_inputs])
    # X_in = W*X + b
        # X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (10 batches, 16 steps, 128 hidden) 换回3维
        X_in = tf.reshape(X, [-1, self.n_steps, self.n_hidden_units])

    # 使用 basic LSTM Cell.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1.0)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        # init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
        # outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
        # 多层LSTM计算过程
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.n_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(X_in[:, time_step, :], init_state)
                outputs.append(cell_output)
        outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden_units])
        # outputs = tf.reshape(outputs, [-1, self.n_hidden_units])
        results = tf.matmul(outputs, weights['out']) + biases['out']
        return results

    def embedding_y(self, label):
            result = []
            for i in label:
                for j in i:
                    z1 = [0]*self.n_classes
                    z1[int(j)] = 1
                    result.append(z1)
            return result
