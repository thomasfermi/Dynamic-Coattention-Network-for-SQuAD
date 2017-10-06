import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import logging
from abstract_model import Qa_model


class DCN_qa_model(Qa_model):
    """This is an implementation of the Dynamic Coattention Network model (https://arxiv.org/abs/1611.01604) for 
    question answering.
    Instead of LSTMs, this implementation uses GRUs.

    The DCN_qa_model class is derived from the class Qa_model. Read the comment under "class Qa_model" in 
    abstract_model.py to get a general idea of the model framework. 
    To understand the DCN model architecture and some variable names in the code (like U,Q,D), you need to read the 
    original paper (https://arxiv.org/abs/1611.01604).
    """

    def add_prediction_and_loss(self):
        coattention_context = self.encode(apply_dropout=True)
        prediction_start, prediction_end, loss = self.dp_decode_HMN(coattention_context, apply_dropout=True,
                                                                    apply_l2_reg=False)
        return prediction_start, prediction_end, loss

    def encode(self, apply_dropout=False):
        """Coattention context encoder as introduced in https://arxiv.org/abs/1611.01604 
        Uses GRUs instead of LSTMs. """

        # Each word is represented by a glove word vector (https://nlp.stanford.edu/projects/glove/)
        self.WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                                   trainable=False)

        # map word index (integer) to word vector (100 dimensional float vector)
        self.embedded_q = tf.nn.embedding_lookup(params=self.WEM, ids=self.q_input_placeholder)
        self.embedded_c = tf.nn.embedding_lookup(params=self.WEM, ids=self.c_input_placeholder)

        rnn_size = self.FLAGS.rnn_state_size
        with tf.variable_scope("rnn", reuse=None):
            cell = tf.contrib.rnn.GRUCell(rnn_size)
            if apply_dropout:
                # TODO add separate dropout placeholder for encoding and decoding. Right now the maximum sets
                # enc_keep_prob to 1 during prediction.
                enc_keep_prob = tf.maximum(tf.constant(self.FLAGS.dropout_encoder), self.dropout_placeholder)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=enc_keep_prob)
            q_sequence_length = tf.reduce_sum(tf.cast(self.q_mask_placeholder, tf.int32), axis=1)
            q_sequence_length = tf.reshape(q_sequence_length, [-1, ])

            q_outputs, q_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_q,
                                                         sequence_length=q_sequence_length, dtype=tf.float32,
                                                         time_major=False)

        Qprime = q_outputs
        q_senti = tf.get_variable("q_senti0", (rnn_size,), dtype=tf.float32)
        q_senti = tf.tile(q_senti, tf.shape(Qprime)[0:1])
        q_senti = tf.reshape(q_senti, (-1, 1, tf.shape(Qprime)[2]))
        Qprime = tf.concat([Qprime, q_senti], axis=1)
        Qprime = tf.transpose(Qprime, [0, 2, 1], name="Qprime")

        # add tanh layer to go from Qprime to Q
        WQ = tf.get_variable("WQ", (self.max_q_length + 1, self.max_q_length + 1),
                             initializer=tf.contrib.layers.xavier_initializer())
        bQ = tf.get_variable("bQ_Bias", shape=(rnn_size, self.max_q_length + 1),
                             initializer=tf.contrib.layers.xavier_initializer())
        Q = tf.einsum('ijk,kl->ijl', Qprime, WQ)
        Q = tf.nn.tanh(Q + bQ, name="Q")

        with tf.variable_scope("rnn", reuse=True):
            c_sequence_length = tf.reduce_sum(tf.cast(self.c_mask_placeholder, tf.int32), axis=1)
            c_sequence_length = tf.reshape(c_sequence_length, [-1, ])
            # use the same RNN cell as for the question input
            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_c,
                                                         sequence_length=c_sequence_length,
                                                         dtype=tf.float32,
                                                         time_major=False)

        D = c_outputs
        c_senti = tf.get_variable("c_senti0", (rnn_size,), dtype=tf.float32)
        c_senti = tf.tile(c_senti, tf.shape(D)[0:1])
        c_senti = tf.reshape(c_senti, (-1, 1, tf.shape(D)[2]))
        D = tf.concat([D, c_senti], axis=1)
        D = tf.transpose(D, [0, 2, 1])
        L = tf.einsum('ijk,ijl->ikl', D, Q)
        AQ = tf.nn.softmax(L)
        AD = tf.nn.softmax(tf.transpose(L, [0, 2, 1]))
        CQ = tf.matmul(D, AQ)
        CD1 = tf.matmul(Q, AD)
        CD2 = tf.matmul(CQ, AD)
        CD = tf.concat([CD1, CD2], axis=1)
        CDprime = tf.concat([CD, D], axis=1)
        CDprime = tf.transpose(CDprime, [0, 2, 1])

        with tf.variable_scope("u_rnn", reuse=False):
            cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
            cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
            if apply_dropout:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=enc_keep_prob)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=enc_keep_prob)

            (cc_fw, cc_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=CDprime,
                                                                sequence_length=c_sequence_length,
                                                                dtype=tf.float32)

        U = tf.concat([cc_fw, cc_bw], axis=2)
        logging.debug("U={}".format(U))
        return U

    def dp_decode_HMN(self, U, pool_size=4, apply_dropout=True, cumulative_loss=True, apply_l2_reg=False):
        """ input: coattention_context U. tensor of shape (batch_size, context_length, arbitrary)
        Implementation of dynamic pointer decoder proposed by Xiong et al. ( https://arxiv.org/abs/1611.01604).

        Some of the implementation details such as the way us is obained from U via tf.gather_nd() are explored on toy 
        data in Experimentation_Notebooks/toy_data_examples_for_tile_map_fn_gather_nd_etc.ipynb"""

        def HMN_func(dim, ps):  # ps=pool size, HMN = highway maxout network
            def func(ut, h, us, ue):
                h_us_ue = tf.concat([h, us, ue], axis=1)
                WD = tf.get_variable(name="WD", shape=(5 * dim, dim), dtype='float32',
                                     initializer=xavier_initializer())
                r = tf.nn.tanh(tf.matmul(h_us_ue, WD))
                ut_r = tf.concat([ut, r], axis=1)
                if apply_dropout:
                    ut_r = tf.nn.dropout(ut_r, keep_prob=self.dropout_placeholder)
                W1 = tf.get_variable(name="W1", shape=(3 * dim, dim, ps), dtype='float32',
                                     initializer=xavier_initializer())
                b1 = tf.get_variable(name="b1_Bias", shape=(dim, ps), dtype='float32',
                                     initializer=tf.zeros_initializer())
                mt1 = tf.einsum('bt,top->bop', ut_r, W1) + b1
                mt1 = tf.reduce_max(mt1, axis=2)
                if apply_dropout:
                    mt1 = tf.nn.dropout(mt1, self.dropout_placeholder)
                W2 = tf.get_variable(name="W2", shape=(dim, dim, ps), dtype='float32',
                                     initializer=xavier_initializer())
                b2 = tf.get_variable(name="b2_Bias", shape=(dim, ps), dtype='float32',
                                     initializer=tf.zeros_initializer())
                mt2 = tf.einsum('bi,ijp->bjp', mt1, W2) + b2
                mt2 = tf.reduce_max(mt2, axis=2)
                mt12 = tf.concat([mt1, mt2], axis=1)
                if apply_dropout:
                    mt12 = tf.nn.dropout(mt12, keep_prob=self.dropout_placeholder)
                W3 = tf.get_variable(name="W3", shape=(2 * dim, 1, ps), dtype='float32',
                                     initializer=xavier_initializer())
                b3 = tf.get_variable(name="b3_Bias", shape=(1, ps), dtype='float32', initializer=tf.zeros_initializer())
                hmn = tf.einsum('bi,ijp->bjp', mt12, W3) + b3
                hmn = tf.reduce_max(hmn, axis=2)
                hmn = tf.reshape(hmn, [-1])
                return hmn

            return func

        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)
        neg = tf.constant([0], dtype=tf.float32)
        neg = tf.tile(neg, [tf.shape(float_mask)[0]])
        neg = tf.reshape(neg, (tf.shape(float_mask)[0], 1))
        float_mask = tf.concat([float_mask, neg], axis=1)
        labels_S = tf.concat([self.labels_placeholderS, tf.cast(neg, tf.int32)], axis=1)
        labels_E = tf.concat([self.labels_placeholderE, tf.cast(neg, tf.int32)], axis=1)
        dim = self.FLAGS.rnn_state_size

        # initialize us and ue as first word in context
        i_start = tf.zeros(shape=(tf.shape(U)[0],), dtype='int32')
        i_end = tf.zeros(shape=(tf.shape(U)[0],), dtype='int32')
        idx = tf.range(0, tf.shape(U)[0], 1)
        s_idx = tf.stack([idx, i_start], axis=1)
        e_idx = tf.stack([idx, i_end], axis=1)
        us = tf.gather_nd(U, s_idx)
        ue = tf.gather_nd(U, e_idx)

        HMN_alpha = HMN_func(dim, pool_size)
        HMN_beta = HMN_func(dim, pool_size)

        alphas, betas = [], []
        h = tf.zeros(shape=(tf.shape(U)[0], dim), dtype='float32', name="h_dpd")  # initial hidden state of RNN
        U_transpose = tf.transpose(U, [1, 0, 2])

        with tf.variable_scope("dpd_RNN"):
            cell = tf.contrib.rnn.GRUCell(dim)
            for time_step in range(3):  # number of time steps can be considered as a hyper parameter
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()

                us_ue = tf.concat([us, ue], axis=1)
                _, h = cell(inputs=us_ue, state=h)

                with tf.variable_scope("alpha_HMN"):
                    if time_step >= 1:
                        tf.get_variable_scope().reuse_variables()
                    alpha = tf.map_fn(lambda ut: HMN_alpha(ut, h, us, ue), U_transpose, dtype=tf.float32)
                    alpha = tf.transpose(alpha, [1, 0]) * float_mask

                i_start = tf.argmax(alpha, 1)
                idx = tf.range(0, tf.shape(U)[0], 1)
                s_idx = tf.stack([idx, tf.cast(i_start, 'int32')], axis=1)
                us = tf.gather_nd(U, s_idx)

                with tf.variable_scope("beta_HMN"):
                    if time_step >= 1:
                        tf.get_variable_scope().reuse_variables()
                    beta = tf.map_fn(lambda ut: HMN_beta(ut, h, us, ue), U_transpose, dtype=tf.float32)
                    beta = tf.transpose(beta, [1, 0]) * float_mask

                i_end = tf.argmax(beta, 1)
                e_idx = tf.stack([idx, tf.cast(i_end, 'int32')], axis=1)
                ue = tf.gather_nd(U, e_idx)

                alphas.append(alpha)
                betas.append(beta)

        if cumulative_loss:
            losses_alpha = [tf.nn.softmax_cross_entropy_with_logits(labels=labels_S, logits=a) for a in
                            alphas]
            losses_alpha = [tf.reduce_mean(x) for x in losses_alpha]
            losses_beta = [tf.nn.softmax_cross_entropy_with_logits(labels=labels_E, logits=b) for b in
                           betas]
            losses_beta = [tf.reduce_mean(x) for x in losses_beta]

            loss = tf.reduce_sum([losses_alpha, losses_beta])
        else:
            cross_entropy_start = tf.nn.softmax_cross_entropy_with_logits(labels=labels_S, logits=alpha,
                                                                          name="cross_entropy_start")
            cross_entropy_end = tf.nn.softmax_cross_entropy_with_logits(labels=labels_E, logits=beta,
                                                                        name="cross_entropy_end")
            loss = tf.reduce_mean(cross_entropy_start) + tf.reduce_mean(cross_entropy_end)

        if apply_l2_reg:
            loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "Bias" not in v.name])
            loss += loss_l2 * self.FLAGS.l2_lambda

        return i_start, i_end, loss
