# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import tensorflow as tf
import functools
import numpy as np
import random
import os

def prelu_func(features, initializer=None, scope=None):  # scope=None
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg
prelu = functools.partial(prelu_func, initializer=tf.constant_initializer(1.0))

# 1.embedding for x,additional features and y; 2. first layer of FC; 3.second layer of FC; 4.softmax
class NERModel(object):
    def __init__(self, config, train_we=True, is_training=False, is_validation=False):
        embedding_size = config.embed_size
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        nature_size = config.nature_size
        label_size = config.label_size
        window_size = config.window_size
        dropout = config.dropout
        l2 = config.l2
        lr = config.lr
        dtype = config.dtype

        # set place holder
        self.vocab_placeholder = tf.placeholder(tf.int32, shape=[None, window_size], name='input')    #X
        self.nature_placeholder = tf.placeholder(tf.int32, shape=[None, window_size], name='nature')  #ADDITIONAL FEATURE
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, window_size], name='target') #Y

        with tf.device('/cpu:0'):
            self.vocab_embedding = tf.get_variable(
                'VocabEmbedding', [vocab_size, embedding_size], trainable=train_we, dtype=dtype)
            self.nature_embedding = tf.get_variable('NatureEmbedding', [nature_size, embedding_size], dtype=dtype)

        vocab_input = tf.gather(self.vocab_embedding, self.vocab_placeholder)    #[None,window_size,embedding_size]
        nature_input = tf.gather(self.nature_embedding, self.nature_placeholder) #[None,window_size,embedding_size]

        #window = tf.add(vocab_input, nature_input)
        window=tf.concat([vocab_input,nature_input], 1) #[None,window_size,embedding_size*2].window = tf.add(vocab_input, nature_input)
        embedding_sizee=embedding_size*2
        window = tf.reshape(window, [-1, window_size * embedding_sizee]) #[None,window_size*embedding_size*2]

        window = tf.nn.dropout(window, dropout) #[None,window_size*embedding_size*2]

        with tf.variable_scope('Layer1'):
            w1 = tf.get_variable('w1', [window_size * embedding_sizee, window_size*hidden_size], dtype=dtype)
            b1 = tf.get_variable('b1', [window_size*hidden_size], dtype=dtype)
            h = prelu(tf.matmul(window, w1) + b1) #[None, window_size*hidden_size]
            tf.add_to_collection('total_loss', l2 * tf.nn.l2_loss(w1))

            h = tf.nn.dropout(h, dropout) ##[None, window_size*hidden_size]
        with tf.variable_scope('Layer2'):
            w2 = tf.get_variable('w2', [window_size*hidden_size, window_size*label_size], dtype=dtype)
            b2 = tf.get_variable('b2', [window_size*label_size], dtype=dtype)
            logits = tf.matmul(h, w2) + b2 #[None, window_size*label_size]
            logits=tf.reshape(logits,[-1,window_size,label_size]) #[None, window_size,label_size]
            tf.add_to_collection('total_loss', l2 * tf.nn.l2_loss(w2))
        self.prediction = tf.argmax(logits,axis=2, name="prediction") #[None, window_size]

        if not is_training:
            return

        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=logits))#.softmax_cross_entropy_with_logits
        tf.add_to_collection('total_loss', cross_entropy)
        self.loss = tf.add_n(tf.get_collection('total_loss'))

        optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

# test started: learn to do name entity recognization(toy task: data generate according to rules from human)
def train():
    class Config():
        embed_size = 100
        hidden_size = 100
        window_size = 3
        dropout = 0.5
        l2 = 0.00001
        lr = 0.00001
        vocab_size = 1000
        nature_size = 100
        label_size =50
        dtype = tf.float32

    train_wv=True
    config=Config()
    is_training=True
    model = NERModel(config=config, train_we=train_wv, is_training=is_training)

    ckpt_dir = 'checkpoint_dmn/dummy_test/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint of DMN.")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        for i in range(5000):
            label_list=get_labels()
            input_vocab = np.array(label_list,dtype=np.int32)
            target_list,nature_list=get_label_from_number(label_list)
            input_nature=np.array(nature_list,dtype=np.int32)
            input_label=np.array(target_list,dtype=np.int32)
            loss,_=sess.run((model.loss,model.train_op),
                                             feed_dict={model.vocab_placeholder:input_vocab,
                                                        model.nature_placeholder:input_nature,
                                                        model.labels_placeholder:input_label})
            if i%300==0:
                save_path = ckpt_dir + "model.ckpt"
                saver.save(sess,save_path,global_step=i)
            print(i,"loss:",loss,)

def predict():
    class Config():
        embed_size = 100
        hidden_size = 100
        window_size = 3
        dropout = 0.5
        l2 = 0.0001
        lr = 0.00001
        vocab_size = 1000
        nature_size = 100
        label_size = 50
        dtype = tf.float32

    train_wv = True
    config = Config()
    is_training = False
    model = NERModel(config=config, train_we=train_wv, is_training=is_training)

    ckpt_dir = 'checkpoint_dmn/dummy_test/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(ckpt_dir + "checkpoint"):
            print("Check point exists. going to restoring variables from checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        for i in range(100):
            label_list = get_labels()
            input_vocab = np.array(label_list, dtype=np.int32)
            target_list, nature_list = get_label_from_number(label_list)
            input_nature = np.array(nature_list, dtype=np.int32)
            prediction = sess.run((model.prediction),
                               feed_dict={model.vocab_placeholder: input_vocab,
                                          model.nature_placeholder: input_nature})
            input_label = np.array(target_list, dtype=np.int32)
            print(i);print( "prediction:", prediction);print("input_label:",input_label)

personal_list=[0]
organization_list=[4,6,8]
location_list=[1,3]
none_list=[2,5,7,9]

personal=11
organization=12
location=13
nonee=14

nature_default=15
nature_even_big=16 #in our experiment,a number is set as nature_even_big if it is even and >=4
nature_odd_small=17  #in our experiment,a number is set as nature odd_small if it is odd and <4
def get_label_from_number(x_list,window_size=3):
    y_big=[[nonee]*window_size]*len(x_list)
    nature_list_big=[[nature_default]*window_size]*len(x_list)
    for k,sub_list in enumerate(x_list):
        y=[nonee]*window_size
        nature_list=[nature_default]*window_size
        for i,e in enumerate(sub_list):
            if e in personal_list:
                y[i]=personal
            elif e in organization_list:
                y[i]=organization
            elif e in location_list:
                y[i]=location
            else:
                y[i]=nonee
            y_big[k]=y

            if  e%2==0:#e>=4 and
                nature_list[i]=nature_even_big
            elif  e%2==1:#e<4 and
                nature_list[i] = nature_odd_small
            nature_list_big[k]=nature_list

    for ii, sub_list in enumerate(y_big):
        sub_list
    return y_big,nature_list_big

def generate_multiple_hot(sub_list,label_size=50):
    multiple_hot=[0]

    return multiple_hot
def get_labels(target_sequence_length=6,window_size=3): # get labels in a window of 3.
    result=[]
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x = random.sample(list, target_sequence_length)
    random.shuffle(x)
    size=len(list)
    for i,e in enumerate(list):
        if window_size+i<size-3:
            result.append(x[i:window_size+i])
    return result

1.train()
#2.predict()