import tensorflow as tf
import logging
from abstract_model import Qa_model


class DCN_qa_model(Qa_model):
    """This is an implementation of the Dynamic Coattention Network model (https://arxiv.org/abs/1611.01604).
    It is work in progress. Right now, a simplified DCN encoder is implemented, which uses GRU instead of LSTMs and 
    doesn't use sentinel vectors yet. 
    There is a simple baseline decoder and a dynamic pointer decoder (dp_decode) which is similar to the one proposed 
    in the above mentioned paper.
    """

    def add_prediction_and_loss(self):
        coattention_context = self.encode()
        prediction_start, prediction_end, loss = self.dp_decode(coattention_context)
        return prediction_start, prediction_end, loss

    def encode(self):
        """Coattention context decoder as specified in https://arxiv.org/abs/1611.01604 
        Simplification: Use GRUs instead of LSTMs. Do not use sentinel vectors. """
        self.WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                                   trainable=False)

        self.embedded_q = tf.nn.embedding_lookup(params=self.WEM, ids=self.q_input_placeholder)
        self.embedded_c = tf.nn.embedding_lookup(params=self.WEM, ids=self.c_input_placeholder)

        rnn_size = self.FLAGS.rnn_state_size
        with tf.variable_scope("rnn", reuse=None):
            cell = tf.contrib.rnn.GRUCell(rnn_size)
            q_sequence_length = tf.reduce_sum(tf.cast(self.q_mask_placeholder, tf.int32), axis=1)
            q_sequence_length = tf.reshape(q_sequence_length, [-1, ])
            c_sequence_length = tf.reduce_sum(tf.cast(self.c_mask_placeholder, tf.int32), axis=1)
            c_sequence_length = tf.reshape(c_sequence_length, [-1, ])

            q_outputs, q_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_q,
                                                         sequence_length=q_sequence_length, dtype=tf.float32,
                                                         time_major=False)

        Qprime = q_outputs
        Qprime = tf.transpose(Qprime, [0, 2, 1], name="Qprime")

        # add tanh layer to go from Qprime to Q
        WQ = tf.get_variable("WQ", (self.max_q_length, self.max_q_length),
                             initializer=tf.contrib.layers.xavier_initializer())
        bQ = tf.get_variable("bQ", shape=(rnn_size, self.max_q_length),
                             initializer=tf.contrib.layers.xavier_initializer())
        Q = tf.einsum('ijk,kl->ijl', Qprime, WQ)
        Q = tf.nn.tanh(Q + bQ, name="Q")

        with tf.variable_scope("rnn", reuse=True):
            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_c,
                                                         sequence_length=c_sequence_length,
                                                         dtype=tf.float32,
                                                         time_major=False)

        D = c_outputs
        D = tf.transpose(D, [0, 2, 1], name="D")
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
            (cc_fw, cc_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=CDprime,
                                                                sequence_length=c_sequence_length,
                                                                dtype=tf.float32)

        coattention_context = tf.concat([cc_fw, cc_bw], axis=2)
        logging.info("coattention_context={}".format(coattention_context))
        return coattention_context

    def encode_bi_rnn(self):
        """ Dirty copy paste of encoder. Uses bidirectional RNNs already for the question and context reading """
        # TODO: check whether this improves performance
        self.WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                                   trainable=False)

        self.embedded_q = tf.nn.embedding_lookup(params=self.WEM, ids=self.q_input_placeholder)
        self.embedded_c = tf.nn.embedding_lookup(params=self.WEM, ids=self.c_input_placeholder)

        rnn_size = self.FLAGS.rnn_state_size
        with tf.variable_scope("rnn", reuse=None):
            cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
            cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
            q_sequence_length = tf.reduce_sum(tf.cast(self.q_mask_placeholder, tf.int32), axis=1)
            q_sequence_length = tf.reshape(q_sequence_length, [-1, ])
            c_sequence_length = tf.reduce_sum(tf.cast(self.c_mask_placeholder, tf.int32), axis=1)
            c_sequence_length = tf.reshape(c_sequence_length, [-1, ])

            (q_outputs_fw, q_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                              inputs=self.embedded_q,
                                                                              sequence_length=q_sequence_length,
                                                                              dtype=tf.float32)

        q_outputs = tf.concat([q_outputs_fw, q_outputs_bw], axis=2)

        Qprime = q_outputs
        Qprime = tf.transpose(Qprime, [0, 2, 1], name="Qprime")

        # add tanh layer to go from Qprime to Q
        WQ = tf.get_variable("WQ", (self.max_q_length, self.max_q_length),
                             initializer=tf.contrib.layers.xavier_initializer())
        bQ = tf.get_variable("bQ", shape=(Qprime.shape[1], self.max_q_length),
                             initializer=tf.contrib.layers.xavier_initializer())
        Q = tf.einsum('ijk,kl->ijl', Qprime, WQ)
        Q = tf.nn.tanh(Q + bQ, name="Q")

        with tf.variable_scope("rnn", reuse=True):
            (c_outputs_fw, c_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                              inputs=self.embedded_c,
                                                                              sequence_length=c_sequence_length,
                                                                              dtype=tf.float32)

        c_outputs = tf.concat([c_outputs_fw, c_outputs_bw], axis=2)

        D = c_outputs
        D = tf.transpose(D, [0, 2, 1], name="D")
        L = tf.einsum('ijk,ijl->ikl', D, Q)
        AQ = tf.nn.softmax(L)  # TODO: is it the right dimension?
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
            (cc_fw, cc_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=CDprime,
                                                                sequence_length=c_sequence_length,
                                                                dtype=tf.float32)

        coattention_context = tf.concat([cc_fw, cc_bw], axis=2)
        logging.info("coattention_context={}".format(coattention_context))
        return coattention_context

    def decode_with_baseline_decoder(self, coattention_context):
        """ input: coattention_context. tensor of shape (batch_size, context_length, arbitrary) 
        Decode via simple projection. condition the end_position on the previously determined start_position. """
        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)

        projector_start = tf.get_variable(name="projectorS", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        projector_end = tf.get_variable(name="projectorE", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())

        prob_start = tf.einsum('ijk,k->ij', coattention_context, projector_start) * float_mask
        prob_end = tf.einsum('ijk,k->ij', coattention_context, projector_end) * float_mask
        prob_start = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(prob_start)
        prob_start = prob_start * float_mask
        prob_end_correlated = tf.concat([prob_end, tf.nn.softmax(prob_start)], axis=1)
        prob_end_correlated = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(prob_end_correlated)
        prob_end_correlated = prob_end_correlated * float_mask

        cross_entropy_start = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholderS,
                                                                      logits=prob_start,
                                                                      name="cross_entropy_start")
        cross_entropy_end = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholderE,
                                                                    logits=prob_end_correlated,
                                                                    name="cross_entropy_end")

        prediction_start = tf.argmax(prob_start, 1)
        prediction_end = tf.argmax(prob_end_correlated, 1)
        loss = tf.reduce_mean(cross_entropy_start) + tf.reduce_mean(cross_entropy_end)
        return prediction_start, prediction_end, loss

    def dp_decode(self, coattention_context):
        """ input: coattention_context. tensor of shape (batch_size, context_length, arbitrary)
        A decoder very similar to the dynamic pointer decoder proposed by Xiong et al. (
        https://arxiv.org/abs/1611.01604). Works as follows:
        1. Project each hidden vector corresponding to a context word onto a weight vector. This results in a vector 
        of length=context_length. Apply softmax and interpret as probability that word i in the context is start 
        word. Use another weight vector to get prob_end.
        2. sum_i prob_start[i] * hidden_context_vector[i] gives a hidden representation of the start hidden vector, 
        which we call start_rep. Similarly we find end_rep.
        3. Use a short RNN (just around 4 timesteps), which is initialized with a hidden state = 0 and does the 
        following updates:
        3 (a) h = cell(in=[start_rep, end_rep], h)
        3 (b) Each context vector gets and appendage: u_c -> [u_c, start_rep, end_rep, h]
        3 (c) Feed all context vectors through a Multilayer Perceptron, obtaining a matrix of the shape that the initial
        coattention matrix had. Consider this as new coattention matrix. Get new start_rep and end_rep as in steps 1 
        and 2. Goto 3 (a)
        """
        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)
        projector_start = tf.get_variable(name="projectorS", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        projector_end = tf.get_variable(name="projectorE", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())

        prob_start = tf.einsum('ijk,k->ij', coattention_context, projector_start) * float_mask
        prob_end = tf.einsum('ijk,k->ij', coattention_context, projector_end) * float_mask

        x = coattention_context  # for abbreviation

        logging.info("prob_start={}".format(prob_start))
        logging.info("prob_end={}".format(prob_end))
        # feed s and e though dynamic pointer decoder

        dim = self.FLAGS.rnn_state_size
        h = tf.zeros(shape=(tf.shape(x)[0], 2 * dim), dtype='float32', name="h_dpd")

        with tf.variable_scope("dpd_RNN"):
            cell = tf.contrib.rnn.GRUCell(2 * dim)
            for time_step in range(4):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                start_rep = tf.einsum('bcd,bc->bd', x, tf.nn.softmax(prob_start))
                end_rep = tf.einsum('bcd,bc->bd', x, tf.nn.softmax(prob_end))

                logging.info("start_rep={}".format(start_rep))

                rep = tf.concat([start_rep, end_rep], axis=1)
                logging.info("rep={}".format(rep))

                h, _ = cell(inputs=rep, state=h)
                logging.info("h={}".format(h))
                rep_h = tf.concat([start_rep, end_rep, h], axis=1)
                logging.info("rep_h={}".format(rep_h))

                rep_h = tf.tile(rep_h, [1, tf.shape(x)[1]])
                logging.info("rep_h={}".format(rep_h))
                rep_h = tf.reshape(rep_h, [tf.shape(x)[0], tf.shape(x)[1], 6 * dim])
                logging.info("rep_h={}".format(rep_h))
                x_dnn = tf.concat([x, rep_h], axis=2)

                logging.info("x_dnn={}".format(x_dnn))

                W = tf.get_variable(name="W1", shape=(8 * dim, 4 * dim), dtype='float32',
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable(name="b1", shape=(4 * dim,), dtype='float32',
                                    initializer=tf.contrib.layers.xavier_initializer())

                logging.info("W={}".format(W))
                logging.info("b={}".format(b))
                x_dnn = tf.nn.relu(tf.einsum('bcd,dn->bcn', x_dnn, W) + b)

                logging.info("x_dnn={}".format(x_dnn))
                # TODO: add dropout
                W2 = tf.get_variable(name="W2", shape=(4 * dim, 2 * dim), dtype='float32',
                                     initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.get_variable(name="b2", shape=(2 * dim,), dtype='float32',
                                     initializer=tf.contrib.layers.xavier_initializer())
                x_dnn = tf.nn.relu(tf.einsum('bcd,dn->bcn', x_dnn, W2) + b2)

                prob_start = tf.einsum('ijk,k->ij', x_dnn, projector_start) * float_mask
                prob_end = tf.einsum('ijk,k->ij', x_dnn, projector_end) * float_mask

                logging.info("x_dnn={}".format(x_dnn))
        # end of dynamic pointing decoder

        cross_entropy_start = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholderS,
                                                                      logits=prob_start,
                                                                      name="cross_entropy_start")
        cross_entropy_end = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholderE,
                                                                    logits=prob_end,
                                                                    name="cross_entropy_end")
        prediction_start = tf.argmax(prob_start, 1)
        prediction_end = tf.argmax(prob_end, 1)

        logging.info("cross_entropy_end={}".format(cross_entropy_end))

        loss = tf.reduce_mean(cross_entropy_start) + tf.reduce_mean(cross_entropy_end)

        logging.info("loss.shape={}".format(loss.shape))
        return prediction_start, prediction_end, loss
