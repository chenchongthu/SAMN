import numpy as np
import tensorflow as tf
import os
import pandas as pd
import scipy.sparse
import math
import pickle

DATA_ROOT = '../data/delicious'

unique_uid = list()
with open(os.path.join(DATA_ROOT, 'unique_uid_sub.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())
unique_sid = list()
with open(os.path.join(DATA_ROOT, 'unique_sid_sub.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_songs = len(unique_sid)
n_users = len(unique_uid)


def load_data(csv_file, shape=(n_users, n_songs)):
    tp = pd.read_csv(csv_file)
    return tp


print n_songs
tp_test = load_data(os.path.join(DATA_ROOT, 'test.csv'))

tp_train = load_data(os.path.join(DATA_ROOT, 'train.csv'))

tp_valid = load_data(os.path.join(DATA_ROOT, 'valid.csv'))


pkl_file = open(os.path.join(DATA_ROOT, 'trust.dic'), 'rb')

tfset = pickle.load(pkl_file)

max_friend = len(tfset[tfset.keys()[0]])

print max_friend


class NMF:
    def __init__(self, user_num, item_num, embedding_size, memory_size, attention_size, max_friend):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_i = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.input_j = tf.placeholder(tf.int32, [None, 1], name='input_jid')
        self.input_uf = tf.placeholder(tf.int32, [None, max_friend], name="input_uf")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.uidW = tf.Variable(tf.random_uniform([user_num + 1, embedding_size], -0.1, 0.1), name="uidW")
        #self.fidW = tf.Variable(tf.random_uniform([user_num + 1, embedding_size], -0.01, 0.1), name="fidW")
        self.iidW = tf.Variable(tf.random_uniform([item_num, embedding_size], -0.1, 0.1), name="iidW")
        self.i_bias = tf.Variable(tf.constant(0.0, shape=[item_num]), name="i_bias")

        self.Key = tf.Variable(tf.random_uniform([embedding_size, memory_size], -0.1, 0.1))
        self.Mem = tf.Variable(tf.random_uniform([memory_size, embedding_size], 1.0, 1.0))

        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.iid = tf.nn.embedding_lookup(self.iidW, self.input_i)
        self.jid = tf.nn.embedding_lookup(self.iidW, self.input_j)
        self.uid = tf.reshape(self.uid, [-1, embedding_size])
        self.iid = tf.reshape(self.iid, [-1, embedding_size])
        self.jid = tf.reshape(self.jid, [-1, embedding_size])

        self.i_b = tf.gather(self.i_bias, self.input_i)

        self.j_b = tf.gather(self.i_bias, self.input_j)

        l2_loss = 0.0

        with tf.name_scope("memory_attention"):

            self.frien_embedding = tf.nn.embedding_lookup(self.uidW, self.input_uf)
            self.frien_num = tf.cast(tf.not_equal(self.input_uf, user_num), 'float32')
            self.frien_embedding = tf.einsum('ab,abc->abc', self.frien_num, self.frien_embedding)

            self.uid_n = tf.nn.l2_normalize(self.uid, 1)

            self.frien_embedding_n = tf.nn.l2_normalize(self.frien_embedding, 2)

            self.cross_friend = tf.einsum('ac,abc->abc', self.uid_n, self.frien_embedding_n)

            self.att_key = tf.einsum('abc,ck->abk', self.cross_friend, self.Key)

            self.att_mem = tf.nn.softmax(self.att_key)
            self.att_mem = tf.einsum('ab,abc->abc', self.frien_num, self.att_mem)

            self.frien_f1 = tf.einsum('abc,ck->abk', self.att_mem, self.Mem)
            self.frien_f2 = tf.multiply(self.frien_f1, self.frien_embedding)


        with tf.name_scope("friend_attention"):
            WA = tf.Variable(
                tf.random_uniform([embedding_size, attention_size], -0.1, 0.1), name='WA')

            BA = tf.Variable(tf.constant(0.0, shape=[attention_size]), name="BA")

            U_omega = tf.Variable(tf.random_uniform([attention_size, 1], -0.1, 0.1))

            self.frien_j = tf.exp(
                tf.einsum('abc,ck->abk', tf.nn.relu(
                    tf.einsum('abc,ck->abk', self.frien_f2, WA) + BA),
                          U_omega))

            self.frien_j = tf.einsum('ab,abc->abc', self.frien_num, self.frien_j)

            self.frien_sum = tf.reduce_sum(self.frien_j, 1, keep_dims=True) + 1e-8

            self.frien_w = tf.div(self.frien_j, self.frien_sum)

            self.friend = tf.reduce_sum(tf.multiply(self.frien_w, self.frien_f2), 1)

            #self.friend = tf.div(tf.reduce_sum(self.frien_f2, 1),
             #                    tf.reduce_sum(self.frien_num, 1, keep_dims=True) + 1e-8)

            l2_loss2 =  tf.nn.l2_loss(WA) + tf.nn.l2_loss(BA)+tf.nn.l2_loss(U_omega)

            self.friend = tf.nn.dropout(self.friend, self.dropout_keep_prob)

        self.user = self.uid + self.friend

        self.dot1 = tf.reduce_sum(tf.multiply(self.user, self.iid), 1, keep_dims=True) + self.i_b
        self.dot2 = tf.reduce_sum(tf.multiply(self.user, self.jid), 1, keep_dims=True) + self.j_b

        self.sub = self.dot1 - self.dot2

        print self.sub

        l2_loss = l2_loss + tf.nn.l2_loss(self.user) + tf.nn.l2_loss(self.iid) + tf.nn.l2_loss(
            self.jid) + tf.nn.l2_loss(self.i_b) + tf.nn.l2_loss(self.j_b)

        loss1 = tf.log(tf.sigmoid(self.sub))

        # loss2 = tf.square(self.dot1 - self.dot2 - 1)
        self.loss = -tf.reduce_sum(loss1) + 0.01 * l2_loss+0.03*l2_loss2
        # self.loss=tf.reduce_sum(loss2)+0.1*l2_loss


        with tf.name_scope('test'):
            # self.all_dot = tf.einsum('ac,bc->abc', self.uid, self.iidW)
            self.all_pre = tf.matmul(self.user, self.iidW, transpose_b=True) + self.i_bias

            print self.all_pre


def train_step(u_batch, i_batch, j_batch, uf_batch):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_j: j_batch,
        deep.input_uf: uf_batch,
        deep.dropout_keep_prob: 1.0,
    }
    _, loss,x = sess.run(
        [train_op, deep.loss,deep.frien_w ],
        feed_dict)
    return loss,x


def dev_step(tset, trset, tfset):
    """
    Evaluates model on a dev set

    """
    user_te = np.array(tset.keys())
    user_te = user_te[:, np.newaxis]

    input_uf = []
    for i in user_te:

        one_use = n_users * np.ones(max_friend)

        if tfset.has_key(i[0]):
            one_use = tfset[i[0]]

        input_uf.append(one_use)
    input_uf = np.array(input_uf)

    ll = int(len(user_te) / 100)

    recall10 = []
    recall20 = []
    recall30=[]
    recall40=[]
    recall50=[]
    ndcg10 = []
    ndcg20=[]
    ndcg30=[]
    ndcg40=[]
    ndcg50=[]


    for batch_num in range(ll):
        start_index = batch_num * 100
        end_index = min((batch_num + 1) * 100, len(user_te))
        u_batch = user_te[start_index:end_index]

        uf_batch = input_uf[start_index:end_index]

        feed_dict = {
            deep.input_u: u_batch,
            deep.input_uf: uf_batch,
            deep.dropout_keep_prob: 1.0,
        }

        pre = sess.run(
            deep.all_pre, feed_dict)

        pre = np.array(pre)

        args = np.argsort(-pre, axis=1)

        recall10.append(recall_k(args, u_batch, tset, trset, 10))
        recall20.append(recall_k(args, u_batch, tset, trset, 20))
        recall30.append(recall_k(args, u_batch, tset, trset, 30))
        recall40.append(recall_k(args, u_batch, tset, trset, 40))
        recall50.append(recall_k(args, u_batch, tset, trset, 50))

        ndcg10.append(ndcg_k(args, u_batch, tset, trset, 10))
        ndcg20.append(ndcg_k(args, u_batch, tset, trset, 20))
        ndcg30.append(ndcg_k(args, u_batch, tset, trset, 30))
        ndcg40.append(ndcg_k(args, u_batch, tset, trset, 40))
        ndcg50.append(ndcg_k(args, u_batch, tset, trset, 50))

    recall10 = np.hstack(recall10)
    recall20 = np.hstack(recall20)
    recall30 = np.hstack(recall30)
    recall40 = np.hstack(recall40)
    recall50 = np.hstack(recall50)
    #recall100 = np.hstack(recall100)
    ndcg10 = np.hstack(ndcg10)
    ndcg20 = np.hstack(ndcg20)
    ndcg30 = np.hstack(ndcg30)
    ndcg40 = np.hstack(ndcg40)
    ndcg50 = np.hstack(ndcg50)

    print np.mean(recall10),np.mean(recall20),np.mean(recall30),np.mean(recall40),np.mean(recall50)
    print np.mean(ndcg10),np.mean(ndcg20),np.mean(ndcg30),np.mean(ndcg40),np.mean(ndcg50)

    return loss


def recall_k(args, u_batch, tset, trset, k):
    recall = []
    for i in range(len(u_batch)):
        acc = 0.0
        if trset.has_key(u_batch[i][0]):
            j = 0
            ks = 0
            while j < k:
                if args[i][ks] in trset[u_batch[i][0]]:
                    ks = ks + 1
                else:
                    if args[i][ks] in tset[u_batch[i][0]]:
                        acc += 1
                    j += 1
                    ks += 1
        else:
            for j in range(k):
                if args[i][j] in tset[u_batch[i][0]]:
                    acc += 1
        sum1 = min(len(tset[u_batch[i][0]]), k)
        recall.append(acc / sum1)

    return recall


def ndcg_k(args, u_batch, tset, trset, k):
    ndc = []
    for i in range(len(u_batch)):
        rels1 = []
        if trset.has_key(u_batch[i][0]):
            j = 0
            ks = 0
            while j < k:
                if args[i][ks] in trset[u_batch[i][0]]:
                    ks = ks + 1
                else:
                    if args[i][ks] in tset[u_batch[i][0]]:
                        rels1.append(1.0)
                    else:
                        rels1.append(0.0)
                    j += 1
                    ks += 1
        else:
            for j in range(k):
                if args[i][j] in tset[u_batch[i][0]]:
                    rels1.append(1.0)
                else:
                    rels1.append(0.0)
        rels2 = []
        for j in tset[u_batch[i][0]]:
            rels2.append(1.0)
        rels1 = np.array(rels1)
        rels2 = np.array(rels2)

        res = ndcg(rels1, rels2, k)
        ndc.append(res)
    return ndc


def getDCG(rels):
    dcg = 0
    for i in range(len(rels)):
        if rels[i] == 1:
            dcg = dcg + (math.log(2) / math.log(i + 2))
    return dcg


def ndcg(rels1, rels2, num):
    dcg = getDCG(rels1)

    rels2 = rels2[0:num]

    idcg = getDCG(rels2)

    return dcg / idcg


def get_train_instances(u_train, i_train, tfset):
    user_input, input_i, input_j, input_uf = [], [], [], []

    pos = {}
    for i in range(len(u_train)):
        if pos.has_key(u_train[i]):
            pos[u_train[i]].append(i_train[i])
        else:
            pos[u_train[i]] = [i_train[i]]

    for i in range(len(u_train)):
        user_input.append(u_train[i])
        input_i.append(i_train[i])

        j = np.random.randint(n_songs)
        while j in pos[u_train[i]]:
            j = np.random.randint(n_songs)
        input_j.append(j)

        one_use = n_users * np.ones(max_friend)

        if tfset.has_key(u_train[i]):
            one_use = tfset[u_train[i]]
        input_uf.append(one_use)

    user_input = np.array(user_input)
    input_i = np.array(input_i)
    input_j = np.array(input_j)
    input_uf = np.array(input_uf)

    user_input = user_input[:, np.newaxis]
    input_i = input_i[:, np.newaxis]
    input_j = input_j[:, np.newaxis]
    return user_input, input_i, input_j, input_uf


if __name__ == '__main__':
    np.random.seed(2018)
    random_seed = 2018

    #tp_test=tp_valid

    

    u_train = np.array(tp_train['uid'], dtype=np.int32)
    i_train = np.array(tp_train['sid'], dtype=np.int32)
    u_test = np.array(tp_test['uid'], dtype=np.int32)
    i_test = np.array(tp_test['sid'], dtype=np.int32)

    shuffle_indices = np.random.permutation(np.arange(len(u_train)))
    u_train = u_train[shuffle_indices]
    i_train = i_train[shuffle_indices]

    tset = {}

    for i in range(len(u_test)):

        if tset.has_key(u_test[i]):
            tset[u_test[i]].append(i_test[i])
        else:
            tset[u_test[i]] = [i_test[i]]

    trset = {}

    for i in range(len(u_train)):

        if trset.has_key(u_train[i]):
            trset[u_train[i]].append(i_train[i])
        else:
            trset[u_train[i]] = [i_train[i]]

    batch_size = 256

    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = NMF(n_users, n_songs, 128, 8, 16, max_friend)
            #optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.05, initial_accumulator_value=1e-8).minimize(
               deep.loss)
            # optimizer=tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.95).minimize(deep.loss)

            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            for epoch in range(205):
                user_train, input_i, input_j, input_uf = get_train_instances(u_train, i_train, tfset)

                shuffle_indices = np.random.permutation(np.arange(len(user_train)))
                input_i = input_i[shuffle_indices]
                user_train = user_train[shuffle_indices]
                input_j = input_j[shuffle_indices]
                input_uf = input_uf[shuffle_indices]

                ll = int(len(user_train) / batch_size)
                loss = 0.0

                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(user_train))

                    u_batch = user_train[start_index:end_index]
                    i_batch = input_i[start_index:end_index]
                    j_batch = input_j[start_index:end_index]

                    uf_batch = input_uf[start_index:end_index]

                    loss1,x = train_step(u_batch, i_batch, j_batch, uf_batch)
                    loss += loss1
                print epoch, loss / ll#,u_batch[0]
                #print x[0][0:20]

                # dev_step(trset,tset)

              
                if epoch<200:
                    if epoch%5==0:
                        dev_step(tset, trset,tfset)

                        #print x.shape
                        #print x[0][0]

                if epoch>=200:
                    dev_step(tset, trset,tfset)























