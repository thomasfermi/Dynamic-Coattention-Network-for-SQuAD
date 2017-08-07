import numpy as np
import tensorflow as tf
from tqdm import trange
import logging
import matplotlib as mpl

mpl.use('Agg')  # can run on machine without display
import matplotlib.pyplot as plt
import os


class Simple_baseline_qa_model(object):
    def __init__(self, max_q_length, max_c_length, data_dir):
        self.max_q_length = max_q_length
        self.max_c_length = max_c_length
        self.data_dir = data_dir + "/"

        self.test_preprocessing_units()
        self.load_and_preprocess_data()

        # self.build_model()

    ####################################################################################################################
    ######################## Loading and preprocessing data ############################################################
    ####################################################################################################################
    def read_and_pad(self, filename, length, pad_value):
        """filename is a file with words ids. Each row can have a different amount of ids.
        Read and pad each row with @pad_value such that it has length @length. 
        Additionally create a boolean mask for each row. An element is False iff the corresponding id is a pad_value
        Returns the padded id array, and the boolean mask."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        lines = [line.split() for line in lines]
        line_array, mask_array = [], []
        for line in lines:
            line = line[:length]
            add_length = length - len(line)
            mask = [True] * len(line) + add_length * [False]
            line = line + add_length * [pad_value]
            line_array.append(line)
            mask_array.append(mask)
        return np.array(line_array, dtype=np.int32), np.array(mask_array)

    def span_to_y(self, y, max_length=None):
        """y is a numpy array, where each row consists of two ints start_id and end_id. 
        Do a one hot encoding of the start_id, and another one for the end_id.
        @max_length is the length of a context paragraph. Hence the one hot vectors have length @max_length"""
        if max_length is None:
            max_length = self.max_c_length
        start_ids, end_ids = y[:, 0], y[:, 1]
        S, E = [], []
        for i in range(len(start_ids)):
            labelS, labelE = np.zeros(max_length, dtype=np.int32), np.zeros(max_length, dtype=np.int32)
            labelS[start_ids[i]], labelE[end_ids[i]] = 1, 1  # one hot encoding
            E.append(labelE)
            S.append(labelS)
        return np.array(S), np.array(E)

    def load_and_preprocess_data(self):
        """Read in the Word embedding matrix as well as the question and context paragraphs and bring them into the 
        desired numerical shape."""

        logging.info("Data prep")
        # load word embedding
        self.WordEmbeddingMatrix = np.load(self.data_dir + "glove.trimmed.100.npz")['glove']
        logging.info("WordEmbeddingMatrix.shape={}".format(self.WordEmbeddingMatrix.shape))
        null_wordvec_index = self.WordEmbeddingMatrix.shape[0]
        # append a zero vector to WordEmbeddingMatrix, which shall be used as padding value
        self.WordEmbeddingMatrix = np.vstack((self.WordEmbeddingMatrix, np.zeros(100)))
        self.WordEmbeddingMatrix = self.WordEmbeddingMatrix.astype(np.float32)
        logging.info("WordEmbeddingMatrix.shape after appending zero vector={}".format(self.WordEmbeddingMatrix.shape))

        self.build_model()

        # load contexts, questions and labels
        self.yS, self.yE = self.span_to_y(np.loadtxt(self.data_dir + "train.span", dtype=np.int32))
        self.yvalS, self.yvalE = self.span_to_y(np.loadtxt(self.data_dir + "val.span", dtype=np.int32))

        self.X_c, self.X_c_mask = self.read_and_pad(self.data_dir + "train.ids.context", self.max_c_length,
                                                    null_wordvec_index)
        self.Xval_c, self.Xval_c_mask = self.read_and_pad(self.data_dir + "val.ids.context", self.max_c_length,
                                                          null_wordvec_index)
        self.X_q, self.X_q_mask = self.read_and_pad(self.data_dir + "train.ids.question", self.max_q_length,
                                                    null_wordvec_index)
        self.Xval_q, self.Xval_q_mask = self.read_and_pad(self.data_dir + "val.ids.question", self.max_q_length,
                                                          null_wordvec_index)
        logging.info("End data prep")

    ####################################################################################################################
    ######################## Model building ############################################################################
    ####################################################################################################################
    def build_model(self):
        self.q_input_placeholder = tf.placeholder(tf.int32, (None, self.max_q_length), name="q_input_ph")
        self.q_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_q_length),
                                                 name="q_mask_placeholder")
        self.c_input_placeholder = tf.placeholder(tf.int32, (None, self.max_c_length), name="c_input_ph")
        self.c_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_c_length),
                                                 name="c_mask_placeholder")
        self.labels_placeholderS = tf.placeholder(tf.int32, (None, self.max_c_length), name="label_phS")
        self.labels_placeholderE = tf.placeholder(tf.int32, (None, self.max_c_length), name="label_phE")

        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_ph")

        self.WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                                   trainable=False)

        self.embedded_q = tf.nn.embedding_lookup(params=self.WEM, ids=self.q_input_placeholder)
        self.embedded_c = tf.nn.embedding_lookup(params=self.WEM, ids=self.c_input_placeholder)

        logging.info("embedded_q.shape={}".format(self.embedded_q.shape))
        logging.info("embedded_c.shape={}".format(self.embedded_c.shape))
        logging.info("labels_placeholderS.shape={}".format(self.labels_placeholderS.shape))

        # RNN magic !
        rnn_size = 64
        with tf.variable_scope("rnn", reuse=None):
            cell = tf.contrib.rnn.GRUCell(rnn_size)
            q_sequence_length = tf.reduce_sum(tf.cast(self.q_mask_placeholder, tf.int32), axis=1)
            q_sequence_length = tf.reshape(q_sequence_length, [-1, ])
            c_sequence_length = tf.reduce_sum(tf.cast(self.c_mask_placeholder, tf.int32), axis=1)
            c_sequence_length = tf.reshape(c_sequence_length, [-1, ])

            q_outputs, q_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_q,
                                                         sequence_length=q_sequence_length, dtype=tf.float32,
                                                         time_major=False)
            question_rep = q_final_state

        with tf.variable_scope("rnn", reuse=True):
            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_c,
                                                         sequence_length=c_sequence_length,
                                                         initial_state=question_rep,
                                                         time_major=False)

        logging.info("Everything went better than expected")

        logging.info("c_outputs.shape={}".format(c_outputs.shape))
        logging.info("q_outputs.shape={}".format(q_outputs.shape))
        logging.info("question_rep.shape={}".format(question_rep.shape))

        attention = tf.einsum('ik,ijk->ij', question_rep, c_outputs)
        logging.info("attention.shape={}".format(attention))
        # weighted_context = tf.einsum('ijk,ij->ijk',c_outputs, attention)
        # logging.info("weighted_context={}".format(weighted_context))
        # knowledge_vector = tf.reshape(weighted_context, [-1,self.max_c_length*rnn_size])
        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)
        knowledge_vector = attention * float_mask
        logging.info("knowledge_vector={}".format(knowledge_vector))

        xe = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(knowledge_vector)
        logging.info("xe.shape={}".format(xe.shape))

        xs = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(knowledge_vector)

        logging.info("self.c_mask_placeholder.shape={}".format(self.c_mask_placeholder.shape))

        int_mask = tf.cast(self.c_mask_placeholder, dtype=tf.int32)
        xs = xs * float_mask
        xe = xe * float_mask
        mls = self.labels_placeholderS * int_mask
        mle = self.labels_placeholderE * int_mask

        cross_entropyS = tf.nn.softmax_cross_entropy_with_logits(labels=mls, logits=xs,
                                                                 name="cross_entropyS")
        cross_entropyE = tf.nn.softmax_cross_entropy_with_logits(labels=mle, logits=xe,
                                                                 name="cross_entropyE")
        self.predictionS = tf.argmax(xs, 1)
        self.predictionE = tf.argmax(xe, 1)

        logging.info("cross_entropyE.shape={}".format(cross_entropyE.shape))

        self.loss = tf.reduce_mean(cross_entropyS) + tf.reduce_mean(cross_entropyE)

        logging.info("loss.shape={}".format(self.loss.shape))

        # use adam optimizer with exponentially decaying learning rate
        # step_adam = tf.Variable(0, trainable=False)
        # rate_adam = tf.train.exponential_decay(1e-3, step_adam, 1, 0.999)  # after one epoch: 0.999**2500 = 0.1
        rate_adam = 1e-3
        # hence learning rate decays by a factor of 0.1 each epoch
        optimizer = tf.train.AdamOptimizer(rate_adam)

        grads_and_vars = optimizer.compute_gradients(self.loss)
        variables = [output[1] for output in grads_and_vars]
        gradients = [output[0] for output in grads_and_vars]

        # gradients = tf.clip_by_global_norm(gradients, clip_norm=1)[0]
        self.global_grad_norm = tf.global_norm(gradients)
        grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]

        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def get_feed_dict(self, batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE):
        feed_dict = {self.c_input_placeholder: batch_xc,
                     self.c_mask_placeholder: batch_xc_mask,
                     self.q_input_placeholder: batch_xq,
                     self.q_mask_placeholder: batch_xq_mask,
                     self.labels_placeholderS: batch_yS,
                     self.labels_placeholderE: batch_yE}
        return feed_dict

    ####################################################################################################################
    ######################## Evaluation metrics and plotting (of those metrics) ########################################
    ####################################################################################################################
    def get_f1(self, yS, yE, ypS, ypE, mask):
        f1_tot = 0.0
        for i in range(len(yS)):
            y = np.zeros(self.max_c_length)
            s = np.argmax(yS[i])
            e = np.argmax(yE[i])
            y[s:e + 1] = 1

            yp = np.zeros_like(y)
            yp[ypS[i]:ypE[i] + 1] = 1
            yp[ypE[i]:ypS[i] + 1] = 1  # allow flipping between start and end

            n_true_pos = np.sum(y * yp)
            n_pred_pos = np.sum(yp)
            n_actual_pos = np.sum(y)
            if n_true_pos == 0:
                f1_tot += 0
            else:
                precision = 1.0 * n_true_pos / n_pred_pos
                recall = 1.0 * n_true_pos / n_actual_pos
                f1_tot += (2 * precision * recall) / (precision + recall)
        f1_tot /= len(yS)
        return f1_tot

    def get_exact_match(self, yS, yE, ypS, ypE, mask):
        count = 0
        for i in range(len(yS)):
            s = np.argmax(yS[i])
            e = np.argmax(yE[i])
            if np.array_equal(s, ypS[i]) and np.array_equal(e, ypE[i]):
                count += 1
        match_fraction = count / float(len(yS))
        return match_fraction

    def plot_metrics(self, epoch_axis, global_losses, global_EMs, global_f1s, global_grad_norms):
        plt.plot(epoch_axis, global_losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig("figs/training_losses_over_time.png")
        plt.close()

        plt.plot(epoch_axis, global_EMs)
        plt.xlabel("epoch")
        plt.ylabel("EM")
        plt.savefig("figs/training_EMs_over_time.png")
        plt.close()

        plt.plot(epoch_axis, global_f1s)
        plt.xlabel("epoch")
        plt.ylabel("F1")
        plt.savefig("figs/training_f1s_over_time.png")
        plt.close()

        plt.plot(epoch_axis, global_grad_norms)
        plt.xlabel("epoch")
        plt.ylabel("gradient_norm")
        plt.savefig("figs/training_grad_norms_over_time.png")
        plt.close()

    def plot_evaluation_metrics(self, EMs_val, F1s_val):
        plt.plot(EMs_val)
        plt.xlabel("epoch")
        plt.ylabel("EM_val")
        plt.savefig("figs/EM_val_over_time.png")
        plt.close()

        plt.plot(F1s_val)
        plt.xlabel("epoch")
        plt.ylabel("F1_val")
        plt.savefig("figs/F1_val_over_time.png")
        plt.close()

    ####################################################################################################################
    ######################## Batch processing ##########################################################################
    ####################################################################################################################
    def initialize_batch_processing(self, n_samples, permutation=None):
        self.batch_index = 0
        self.max_batch_index = n_samples
        if permutation == 'by_length':
            # sum over True/False gives number of words in each sample
            length_of_each_context_paragraph = np.sum(self.X_c_mask, axis=1)
            # permutation of data is chosen, such that the algorithm sees short context_paragraphs first
            self.batch_permutation = np.argsort(length_of_each_context_paragraph)
        elif permutation == 'random':
            self.batch_permutation = np.random.permutation(self.max_batch_index)  # random initial permutation
        else:  # no permutation
            self.batch_permutation = np.arange(self.max_batch_index)  # initial permutation = identity

    def next_batch(self, batch_size):
        if self.batch_index >= self.max_batch_index:
            self.batch_index = 0
            self.batch_permutation = np.random.permutation(self.max_batch_index)

        start = self.batch_index
        end = self.batch_index + batch_size

        Xcres = self.X_c[self.batch_permutation[start:end]]
        Xcmaskres = self.X_c_mask[self.batch_permutation[start:end]]
        Xqres = self.X_q[self.batch_permutation[start:end]]
        Xqmaskres = self.X_q_mask[self.batch_permutation[start:end]]
        yresS = self.yS[self.batch_permutation[start:end]]
        yresE = self.yE[self.batch_permutation[start:end]]

        # Xcres = self.X_c[start:end]
        # Xcmaskres = self.X_c_mask[start:end]
        # Xqres = self.X_q[start:end]
        # Xqmaskres = self.X_q_mask[start:end]
        # yresS = self.yS[start:end]
        # yresE = self.yE[start:end]

        self.batch_index += batch_size
        return Xcres, Xcmaskres, Xqres, Xqmaskres, yresS, yresE

    ####################################################################################################################
    ######################## Unit tests ################################################################################
    ####################################################################################################################
    def test_preprocessing_units(self):
        ################## test for span_to_y ##################
        y = np.array([[1, 2], [2, 4], [1, 1], [0, 0]], dtype=np.int32)
        yS, yE = self.span_to_y(y, 5)
        assert np.array_equal(yS[0], np.array([0, 1, 0, 0, 0], dtype=np.int32))
        assert np.array_equal(yE[0], np.array([0, 0, 1, 0, 0], dtype=np.int32))

        assert np.array_equal(yS[1], np.array([0, 0, 1, 0, 0], dtype=np.int32))
        assert np.array_equal(yE[1], np.array([0, 0, 0, 0, 1], dtype=np.int32))

        assert np.array_equal(yS[2], np.array([0, 1, 0, 0, 0], dtype=np.int32))
        assert np.array_equal(yE[2], np.array([0, 1, 0, 0, 0], dtype=np.int32))

        assert np.array_equal(yS[3], np.array([1, 0, 0, 0, 0], dtype=np.int32))
        assert np.array_equal(yE[3], np.array([1, 0, 0, 0, 0], dtype=np.int32))
        logging.info("span_to_y passed the test")

        ################## test for read and pad ##################
        filename = "unit_test_train.ids.context"
        with open(filename, 'w') as f:
            f.write("0 1 2\n")
            f.write("0 1 0 1 0 1\n")
            f.write("2 1\n")
        length = 5
        pad_value = -1
        c, c_mask = self.read_and_pad(filename, length, pad_value)
        c_as_should_be = np.array([[0, 1, 2, -1, -1], [0, 1, 0, 1, 0], [2, 1, -1, -1, -1]], dtype=np.int32)
        c_mask_as_should_be = np.array([[True, True, True, False, False],
                                        [True, True, True, True, True],
                                        [True, True, False, False, False]])

        assert np.array_equal(c, c_as_should_be)
        assert np.array_equal(c_mask, c_mask_as_should_be)
        os.remove(filename)
        logging.info("read_and_pad passed the test")

    ####################################################################################################################
    ######################## Training ##################################################################################
    ####################################################################################################################
    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        epochs = 8
        batch_size = 32
        n_samples = len(self.yS)
        self.initialize_batch_processing(n_samples=n_samples)

        global_losses, global_EMs, global_f1s, global_grad_norms = [], [], [], []  # global means "over several epochs"
        EMs_val, F1s_val = [], []  # exact_match- and F1-metrics on the validation data

        for index_epoch in range(1, epochs + 1):
            progbar = trange(int(n_samples / batch_size))
            losses, EMs, f1s, grad_norms = [], [], [], []

            ############### train for one epoch ###############
            for _ in progbar:
                batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE = self.next_batch(
                    batch_size=batch_size)
                feed_dict = self.get_feed_dict(batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE)
                _, current_loss, predictionS, predictionE, grad_norm = sess.run(
                    [self.train_op, self.loss, self.predictionS, self.predictionE, self.global_grad_norm],
                    feed_dict=feed_dict)
                EMs.append(self.get_exact_match(batch_yS, batch_yE, predictionS, predictionE, batch_xc_mask))
                f1s.append(self.get_f1(batch_yS, batch_yE, predictionS, predictionE, batch_xc_mask))
                losses.append(current_loss)
                grad_norms.append(grad_norm)

                if len(losses) >= 20:
                    progbar.set_postfix({'loss': np.mean(losses), 'EM': np.mean(EMs), 'f1': np.mean(f1s),
                                         'grad_norm': np.mean(grad_norms)})
                    global_losses.append(np.mean(losses))
                    global_EMs.append(np.mean(EMs))
                    global_f1s.append(np.mean(f1s))
                    global_grad_norms.append(np.mean(grad_norms))
                    losses, EMs, f1s, grad_norms = [], [], [], []

            ############### After an epoch: evaluate on validation set ###############
            logging.info("Epoch {} finished. Doing evaluation on validation set...".format(index_epoch))
            feed_dict = self.get_feed_dict(self.Xval_c, self.Xval_c_mask, self.Xval_q, self.Xval_q_mask, self.yvalS,
                                           self.yvalE)
            val_loss, predictionS, predictionE = sess.run([self.loss, self.predictionS, self.predictionE],
                                                          feed_dict=feed_dict)

            EM_val = self.get_exact_match(self.yvalS, self.yvalE, predictionS, predictionE, self.Xval_c_mask)
            F1_val = self.get_f1(self.yvalS, self.yvalE, predictionS, predictionE, self.Xval_c_mask)

            logging.info("EM_val={}".format(EM_val))
            logging.info("F1_val={}".format(F1_val))
            EMs_val.append(EM_val)
            F1s_val.append(F1_val)

            ############### do some plotting ###############
            n_data_points = len(global_losses)
            epoch_axis = np.arange(n_data_points, dtype=np.float32) * index_epoch / float(n_data_points)
            self.plot_metrics(epoch_axis, global_losses, global_EMs, global_f1s, global_grad_norms)

        # after all epochs have finished. plot the evaluation metrics
        self.plot_evaluation_metrics(EMs_val, F1s_val)
