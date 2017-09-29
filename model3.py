import tensorflow as tf
import numpy as np
import random

#用于将打印完全显示
np.set_printoptions(threshold=np.inf)

class Model(object):
    def __init__(self, id2, vocab_size, class_num, data_train, data_test):

    # hyperparameters
        self.lr = 0.001                  # learning rate
        self.training_iters = 9011     # train step 上限
        self.testing_iters = 20         # test 病例上限
        self.batch_size = 10            #训练批次大小
        self.batch_test_size = 1        #测试病例选取参数
        self.vocab_size = vocab_size   #样本中不同字的个数，处理数据的时候得到
        self.n_inputs = self.embedding_size = 64  # MNIST data input 
        self.n_steps = 24                # time steps
        self.n_hidden_units = 128        # neurons in hidden layer
        self.n_classes = class_num              # MNIST classes (0-18 digits)
        
        self.id2word, self.id2tag = id2      #标号解码
        
    # placeholder
        x = tf.placeholder(tf.int32, [None, self.n_steps])
        y = tf.placeholder(tf.float32, [None,self.n_classes])
        batch_sizes = tf.placeholder(tf.int32,[]) 

    # 对 weights biases 初始值的定义
        weights = {
    # shape (64, 128) 
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
    # shape (128*2, class_num)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units*2, self.n_classes]))
        }
        biases = {
    # shape (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
    # shape (class_num, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }
    #获取预测结果，计算损失函数，梯度下降
        pred = self.lstm(x, weights, biases,batch_sizes)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(self.lr).minimize(cost)
    #按字的正确率统计
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print('Finished creating the lstm model.')

    #initial 
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            step = 0
            step2 = 0
    #训练
            print("Traing")
            while step * self.batch_size < self.training_iters:
                batch_xs, batch_ys = data_train.next_batch(self.batch_size)
                batch_y = np.asarray(self.embedding_y(batch_ys))
                feed_dict={
                    x: batch_xs,
                    y: batch_y,
                    batch_sizes:self.batch_size,
                }
    #启动训练
                sess.run([train_op], feed_dict=feed_dict)
                #if step % 100 == 0:
                    #print(sess.run(accuracy,feed_dict=feed_dict))
                    #print(sess.run(tf.argmax(pred,1),feed_dict=feed_dict))
                    #print(sess.run(tf.argmax(y,1),feed_dict=feed_dict))
                step += 1
    #测试
            print("Training is over,start testing")
            while step2 * self.batch_test_size<self.testing_iters:
                batch_x_test, batch_y_test = data_test.next_batch(self.batch_test_size)
                batch_x_test = np.asarray(batch_x_test[0])
                batch_yt = np.asarray(self.embedding_y(batch_y_test[0]))
                l=len(batch_x_test)
                feed_dict={
                    x: batch_x_test,
                    y: batch_yt,
                    batch_sizes: l,
                }
                if step2 == 1:
                    xx = self.combine(batch_x_test.reshape(-1))
                    word = self.id2word[xx]
                    pred_result = self.combine(sess.run(tf.argmax(pred, 1), feed_dict=feed_dict))
                    y_result = self.combine(sess.run(tf.argmax(y, 1), feed_dict=feed_dict))
                    pred_tag = self.id2tag[pred_result]
                    y_tag = self.id2tag[y_result]
                    for index, ii in enumerate(word):
                        print(ii, pred_tag.values[index], y_tag.values[index])
                    pred_result, result_irregular = self.extract_non(word.values, pred_tag.values)
                    label_result, _ = self.extract_non(word.values, y_tag.values)
                    result_error = [temp for temp in pred_result if temp not in label_result]
                    result_correct = [temp for temp in pred_result if temp in label_result]
                    print('不规则的:', result_irregular)
                    print('标记规则正确但是标记错误:', result_error)
                    print('标记正确:', result_correct)
                    # print(sess.run(tf.argmax(pred, 1), feed_dict=feed_dict))
                    # print(sess.run(tf.argmax(y,1),feed_dict=feed_dict))
                    # print(len(sess.run(tf.argmax(pred,1),feed_dict=feed_dict)))
                    # print(yy)
                    # print(yyy)

                step2 += 1
                
    #lstm模型，参数：输入，权重，偏置，批次大小
    def lstm(self, X, weights, biases,batch_sizes):
    # X == (10 batches , 24 steps)
    # X ==> (10 batches * 64 inputs, 24 steps)
        embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
        X = tf.nn.embedding_lookup(embedding, X)
        X = tf.cast(X,tf.float32)
    #X ==> (10 batches * 24 steps, 64 inputs)
        X = tf.reshape(X,[-1,self.n_inputs])
    # X_in = W*X + b
        X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (10 batches, 24 steps, 128 hidden) 换回3维
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
    # 使用 双向 basic LSTM Cell.
        fw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 初始化全零 state 
        fw_init_state = fw_cell.zero_state(batch_sizes, dtype=tf.float32) 
        bw_init_state = bw_cell.zero_state(batch_sizes, dtype=tf.float32)
        
        outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, X_in, initial_state_fw=fw_init_state,
                                                            initial_state_bw=bw_init_state, dtype=tf.float32)
        fw_out, bw_out = outputs
        outputs = tf.concat([fw_out, bw_out], axis=2)
        outputs = tf.reshape(outputs, [-1, self.n_hidden_units*2])
        results = tf.matmul(outputs, weights['out']) + biases['out']
        
        return results
    
    # 合并函数
    def combine(self,y):
        r=[]
        result=[]
        i=0
        while i<len(y):
            r.append(y[i:i+24])
            i+=24
    # 计算原病例长度
        l=8*(len(r)-1)+24
    # 开始合并
        if l==24:
            rerult=r[0]
        else:
            for ii in range(l):
    # 前8位无重叠
                if ii<8:
                    result.append(r[0][ii])
    # 8-16位有两组相同
                elif ii<16:
                    m=[r[int(ii/8)-1][8+int(ii%8)],r[int(ii/8)][int(ii%8)]]
                    if m[0] == m[1]:
                       result.append(m[0])
                    else:
                        if m[0]*m[1] !=0:
                            result.append(m[random.randint(0,1)])
                        else:
                            if m[0]!=0:
                                result.append(m[0])
                            else:
                                result.append(m[1])
    # 16-(l-16)有三组相同
                elif ii<l-16:
                    n=[r[int(ii/8)-2][16+int(ii%8)],r[int(ii/8)-1][8+int(ii%8)],r[int(ii/8)][int(ii%8)]]
                    if n[0]*n[1]*n[2]!=0:
                        if n[0]==n[1]:
                            result.append(n[0])
                        elif n[1]==n[2]:
                            result.append(n[1])
                        elif n[0]==n[2]:
                            result.append(n[0])
                        else:
                            result.append(n[random.randint(0,2)])
                    else:
                        if n[0]==0:
                            if n[1]==0:
                                result.append(n[2])
                            else:
                                if n[2]==0:
                                    result.append(n[1])
                                else:
                                    result.append(n[random.randint(1,2)])
                        else:
                            if n[1]==0:
                                if n[2]==0:
                                    result.append(n[0])
                                else:
                                    result.append(n[2*random.randint(0,1)])
                            else:
                                result.append(n[random.randint(0,1)])
    # (l-16)-(l-8)有两组相同
                elif ii<l-8:
                    m=[r[int(ii/8)-2][16+int(ii%8)], r[int(ii/8)-1][8+int(ii%8)]]
                    if m[0] == m[1]:
                        result.append(m[0])
                    else:
                        if m[0]*m[1] !=0:
                            result.append(m[random.randint(0,1)])
                        else:
                            if m[0]!=0:
                                result.append(m[0])
                            else:
                                result.append(m[1])
    # (l-8)-l无重叠
                else:
                    result.append(r[int(ii/8)-2][16+int(ii%8)])
        return result

    # label 的one-hot
    def embedding_y(self,label):
            result = []
            for i in label:
                for j in i:
                    z1 = [0]*self.n_classes
                    z1[int(j)]=1
                    result.append(z1)
            return result

    def extract_non(self, words, labels):
        i = 0
        words, labels = list(words), list(labels)
        temp, result, index_correct, result_irregular = [], [], [], []
        while i < len(labels):
            if labels[i][0] == 'S':
                result.append([words[i], labels[i], i])
                index_correct.append(i)
                i += 1
            elif labels[i][0] == 'B':
                k = i + 1
                temp += [labels[i][0], labels[k][0]]
                while labels[k] != 'O' and labels[k][-3:] == labels[i][-3:] and labels[k][0] not in ['E', 'S', 'B']:
                    k += 1
                    if k < len(labels):
                        temp += labels[k][0]
                    else:
                        break
                if temp == ['B'] + ['M'] * (len(temp) - 2) + ['E'] or temp == ['B', 'E']:
                    result.append([words[i:k + 1], labels[i:k+1], i, k])
                    index_correct.extend(list(range(i, k+1)))
                temp = []
                if labels[k][0] in ['B', 'S']:
                    i = k
                else:
                    i = k + 1
            else:
                i += 1
        ii = 0
        while ii < len(labels):
            if ii not in index_correct and labels[ii] != 'O':
                forwad, backward = ii, ii
                fw_temp, bw_temp = 3, 3
                while forwad - 1 >= 0 and forwad - 1 not in index_correct:
                    forwad -= 1
                    if labels[forwad] != 'O' and labels[forwad][-3:] == labels[ii][-3:]:
                        fw_temp += 1
                    if ii - forwad > fw_temp:
                        break
                while backward + 1 < len(labels) and backward + 1 not in index_correct:
                    backward += 1
                    if labels[backward] != 'O' and labels[backward][-3:] == labels[ii][-3:]:
                        bw_temp += 1
                    if backward - ii > bw_temp:
                        break
                result_irregular.append([words[forwad:backward + 1], labels[forwad:backward + 1]])
                ii = backward + 1
            else:
                ii += 1
        return result, result_irregular