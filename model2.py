import tensorflow as tf
import numpy as np
tf.set_random_seed(1)   # set random seed

# 导入数据
np.set_printoptions(threshold=np.inf) 
class Model(object):
    def __init__(self, id2, vocab_size, class_num, data_train, data_test):

        # hyperparameters
        self.lr = 0.001                  # learning rate
        self.training_iters = 9011     # train step 上限
        self.testing_iters = 20
        self.batch_size = 10
        self.batch_test_size = 1
        self.vocab_size = vocab_size   #样本中不同字的个数，处理数据的时候得到
        self.n_inputs = self.embedding_size = 64  # MNIST data input (img shape: 28*28)
        self.n_steps = 16                # time steps
        self.n_hidden_units = 128        # neurons in hidden layer
        self.n_classes = class_num              # MNIST classes (0-9 digits)

        # id2word id2tag
        self.id2word, self.id2tag = id2
# x y placeholder
        x = tf.placeholder(tf.int32, [None, self.n_steps])
        y = tf.placeholder(tf.float32, [None, self.n_classes])
        batch_sizes = tf.placeholder(tf.int32, [])

# 对 weights biases 初始值的定义
        weights = {
    # shape (64, 128) 
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
    # shape (128, class_num)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        biases = {
    # shape (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
    # shape (class_num, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }
        pred = self.RNN(x, weights, biases, batch_sizes)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# 替换成下面的写法:
        print('Finished creating the lstm model.')
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            step = 0
            step2 = 0
            while step * self.batch_size < self.training_iters:
                batch_xs, batch_ys = data_train.next_batch(self.batch_size)
                batch_y = np.asarray(self.embedding_y(batch_ys))
                sess.run([train_op], feed_dict={
                    x: batch_xs,
                    y: batch_y,
                    batch_sizes: self.batch_size,
                })
                if step % 100 == 0:
                    print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_y, batch_sizes:self.batch_size, }))
                    train_pred = sess.run(tf.argmax(pred, -1), feed_dict={x: batch_xs, y: batch_y,
                                                                          batch_sizes: self.batch_size, })
#                    print(sess.run(tf.argmax(y,-1),feed_dict={
#                        x: batch_xs,
#                        y: batch_y,
#                        batch_sizes:self.batch_size,
#                    }))
#                    if step == 200 :
#                        #print(batch_y)
#                        print(sess.run(tf.argmax(pred,-1),feed_dict={
#                        x: batch_xs,
#                        y: batch_y,
#                    }))
#                        print(sess.run(tf.argmax(y,-1),feed_dict={
#                        x: batch_xs,
#                        y: batch_y,
#                    }))
#                    print(sess.run(pred[0],feed_dict={
#                    x: batch_xs,
#                    y: batch_y,
#                }))
                step += 1
            print("training is over,start testing")
            while step2 * self.batch_test_size < self.testing_iters:
                batch_x_test, batch_y_test = data_test.next_batch(self.batch_test_size)
                batch_x_test = np.asarray(batch_x_test[0])
                try:
                    batch_yt = np.asarray(self.embedding_y(batch_y_test[0]))
                except TypeError:
                    batch_yt = np.asarray(self.embedding_y_use(batch_y_test[0]))
                length = len(batch_x_test)
                if step2 % 10 == 0:
                    print(sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_yt, batch_sizes: length, }))
                    test_pred = sess.run(tf.argmax(pred, -1), feed_dict={x: batch_x_test, y: batch_yt,
                                                                         batch_sizes: length, })
                    test_pred = self.combine(test_pred)
                    test_word = self.fuser(batch_x_test)
                    word, tag = self.id2word[test_word], self.id2tag[test_pred]
                    print(''.join(str(k) for k in word.values), tag.values)
#                    print(sess.run(tf.argmax(y,-1),feed_dict={
#                    x: batch_x_test,
#                    y: batch_yt,
#                    batch_sizes:len(batch_x_test[0]),
#                }))
#                if step2==10:
#                    print(batch_x_test)
#                    print(type(len(batch_x_test[0])))
#                    print(batch_yt)
                step2 += 1

    def RNN(self, X, weights, biases,batch_sizes):
        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
        X = tf.nn.embedding_lookup(embedding, X)
    # X ==> (10 batches * 16 steps, 64 inputs)
        X = tf.cast(X, tf.float32)
        X = tf.reshape(X, [-1, self.n_inputs])
    # X_in = W*X + b
        X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (10 batches, 16 steps, 128 hidden) 换回3维
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
    # 使用 basic LSTM Cell.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(batch_sizes, dtype=tf.float32) # 初始化全零 state
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
        outputs = tf.reshape(outputs, [-1, self.n_hidden_units])
        results = tf.matmul(outputs, weights['out']) + biases['out']
        results = tf.reshape(results, [-1, self.n_classes])
        return results

    def embedding_y(self, label):
            result = []
            for i in label:
                for j in i:
                    z1 = [0]*self.n_classes
                    z1[int(j)]=1
                    result.append(z1)
            return result

    def embedding_y_use(self, label):
            result = []
            for i in label:
                temp = [0]*self.n_classes
                temp[int(i)] = 1
                result.append(temp)
            return result

    def fuser(self, inputs):
        result = []
        for index, ii in enumerate(inputs):
            if index == 0:
                result += list(ii)
            else:
                result += list(ii)[8:]
        while result[-1] == 0:
            result.pop()
        return result

    def combine(self, y):
        result = []
        i = 0
        while i < len(y):
            if i < 8:
                result.append(y[i])
                i += 1
            elif i < len(y) - 8:
                if y[i] == y[i + 8]:
                    result.append(y[i])
                    i += 1
                else:
                    if (y[i] * y[i + 8] == 0) and (y[i] + y[i + 8] != 0):
                        if y[i] != 0:
                            result.append(y[i])
                        else:
                            result.append(y[i + 8])
                        i += 1
                    elif (y[i] * y[i + 8] != 0):
                        result.append(y[i])
                        i += 1
                if (i % 8 == 0) and (i + 8 < len(y)):
                    i += 8
            else:
                result.append(y[i])
                i += 1
        while result[-1] == 0:
            result.pop()
        return result