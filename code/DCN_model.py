import tensorflow as tf
import logging
from abstract_model import Qa_model


class DCN_qa_model(Qa_model):
    """This is an implementation of the Dynamic Coattention Network model (https://arxiv.org/abs/1611.01604).
    It is work in progress. Right now, a simplified DCN encoder is implemented, which uses GRU instead of LSTMs and 
    doesn't use sentinel vectors yet. 
    Instead of a decoder from the above paper, there are two very simple baseline decoders implemented.
    """

    def add_prediction_and_loss(self):
        coattention_context = self.encode()
        # prediction_start, prediction_end, loss = self.decode_with_baseline_decoder1(coattention_context)
        prediction_start, prediction_end, loss = self.decode_with_baseline_decoder3(coattention_context)

        return prediction_start, prediction_end, loss

    def encode(self):
        self.WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                                   trainable=False)

        self.embedded_q = tf.nn.embedding_lookup(params=self.WEM, ids=self.q_input_placeholder)
        self.embedded_c = tf.nn.embedding_lookup(params=self.WEM, ids=self.c_input_placeholder)

        logging.info("embedded_q.shape={}".format(self.embedded_q.shape))
        logging.info("embedded_c.shape={}".format(self.embedded_c.shape))
        logging.info("labels_placeholderS.shape={}".format(self.labels_placeholderS.shape))

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
        logging.info("Qprime={}".format(Qprime))

        # add tanh layer to go from Qprime to Q
        WQ = tf.get_variable("WQ", (self.max_q_length, self.max_q_length),
                             initializer=tf.contrib.layers.xavier_initializer())
        bQ = tf.get_variable("bQ", shape=(rnn_size, self.max_q_length),
                             initializer=tf.contrib.layers.xavier_initializer())
        logging.info("WQ={}".format(WQ))
        Q = tf.einsum('ijk,kl->ijl', Qprime, WQ)
        Q = tf.nn.tanh(Q + bQ, name="Q")
        logging.info("Q={}".format(Q))

        with tf.variable_scope("rnn", reuse=True):
            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.embedded_c,
                                                         sequence_length=c_sequence_length,
                                                         dtype=tf.float32,
                                                         time_major=False)

        D = c_outputs
        D = tf.transpose(D, [0, 2, 1], name="D")
        logging.info("D={}".format(D))

        L = tf.einsum('ijk,ijl->ikl', D, Q)
        logging.info("L={}".format(L))

        AQ = tf.nn.softmax(L)  # TODO: is it the right dimension?
        logging.info("AQ={}".format(AQ))
        AD = tf.nn.softmax(tf.transpose(L, [0, 2, 1]))
        logging.info("AD={}".format(AD))

        CQ = tf.matmul(D, AQ)
        logging.info("CQ={}".format(CQ))
        CD1 = tf.matmul(Q, AD)
        CD2 = tf.matmul(CQ, AD)
        CD = tf.concat([CD1, CD2], axis=1)
        CDprime = tf.concat([CD, D], axis=1)
        logging.info("CD1={}".format(CD1))
        logging.info("CD2={}".format(CD2))
        logging.info("CD={}".format(CD))
        logging.info("CDprime={}".format(CDprime))
        CDprime = tf.transpose(CDprime, [0, 2, 1])

        with tf.variable_scope("u_rnn", reuse=False):
            cell = tf.contrib.rnn.GRUCell(2 * rnn_size)
            coattention_context, _ = tf.nn.dynamic_rnn(cell, inputs=CDprime, dtype=tf.float32,
                                                       sequence_length=c_sequence_length)

        logging.info("coattention_context.shape={}".format(coattention_context.shape))

        return coattention_context

    def decode_with_baseline_decoder1(self, coattention_context):
        """ input: coattention_context. tensor of shape (batch_size, context_length, arbitrary) 
        Decoding is done by a simple projection. """
        projector = tf.get_variable(name="projector", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        knowledge_vector = tf.einsum('ijk,k->ij', coattention_context, projector)
        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)
        knowledge_vector = knowledge_vector * float_mask

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

        cross_entropy_start = tf.nn.softmax_cross_entropy_with_logits(labels=mls, logits=xs,
                                                                      name="cross_entropy_start")
        cross_entropy_end = tf.nn.softmax_cross_entropy_with_logits(labels=mle, logits=xe,
                                                                    name="cross_entropy_end")
        prediction_start = tf.argmax(xs, 1)
        prediction_end = tf.argmax(xe, 1)

        logging.info("cross_entropy_end.shape={}".format(cross_entropy_end.shape))

        loss = tf.reduce_mean(cross_entropy_start) + tf.reduce_mean(cross_entropy_end)

        logging.info("loss.shape={}".format(loss.shape))
        return prediction_start, prediction_end, loss

    def decode_with_baseline_decoder2(self, coattention_context):
        """ input: coattention_context. tensor of shape (batch_size, context_length, arbitrary) 
        Advance over baseline_decoder1: First decode prob_start. Then Decode prob_end conditioned on prob_start. """
        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)
        int_mask = tf.cast(self.c_mask_placeholder, dtype=tf.int32)

        coattention_context = tf.nn.dropout(coattention_context, self.dropout_placeholder)

        projector_start = tf.get_variable(name="projectorS", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        projector_end = tf.get_variable(name="projectorE", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())

        prob_start = tf.einsum('ijk,k->ij', coattention_context, projector_start) * float_mask
        prob_end = tf.einsum('ijk,k->ij', coattention_context, projector_end) * float_mask

        logging.info("prob_start={}".format(prob_start))
        prob_start = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(prob_start)
        prob_start = prob_start * float_mask
        logging.info("prob_start={}".format(prob_start))

        prob_end_correlated = tf.concat([prob_end, tf.nn.softmax(prob_start)], axis=1)
        logging.info("prob_end_correlated={}".format(prob_end_correlated))
        # prob_end_correlated = tf.contrib.keras.layers.Dense(self.max_c_length, activation='relu')(prob_end_correlated)
        prob_end_correlated = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(prob_end_correlated)
        logging.info("prob_end_correlated={}".format(prob_end_correlated))

        prob_end_correlated = prob_end_correlated * float_mask

        masked_label_start = self.labels_placeholderS * int_mask
        masked_label_end = self.labels_placeholderE * int_mask

        cross_entropy_start = tf.nn.softmax_cross_entropy_with_logits(labels=masked_label_start, logits=prob_start,
                                                                      name="cross_entropy_start")
        cross_entropy_end = tf.nn.softmax_cross_entropy_with_logits(labels=masked_label_end, logits=prob_end_correlated,
                                                                    name="cross_entropy_end")
        prediction_start = tf.argmax(prob_start, 1)
        prediction_end = tf.argmax(prob_end_correlated, 1)

        logging.info("cross_entropy_end={}".format(cross_entropy_end))

        loss = tf.reduce_mean(cross_entropy_start) + tf.reduce_mean(cross_entropy_end)

        logging.info("loss.shape={}".format(loss.shape))
        return prediction_start, prediction_end, loss

    def decode_with_baseline_decoder3(self, coattention_context):
        """ input: coattention_context. tensor of shape (batch_size, context_length, arbitrary) 
        Advance over baseline_decoder1: First decode prob_start. Then Decode prob_end conditioned on prob_start. """
        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)
        int_mask = tf.cast(self.c_mask_placeholder, dtype=tf.int32)

        coattention_context = tf.nn.dropout(coattention_context, self.dropout_placeholder)

        # feed coattention context through DNN
        def get_weight_and_bias(l):
            l = str(l)
            U = tf.get_variable(name="DNN_U" + l, shape=(coattention_context.shape[2], coattention_context.shape[2]),
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="DNN_b" + l, shape=(coattention_context.shape[1], coattention_context.shape[2]),
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            return U, b

        nlayers = 3
        for layer_index in range(nlayers):
            U, b = get_weight_and_bias(layer_index)
            coattention_context = tf.einsum('ijk,kl->ijl', coattention_context, U)
            coattention_context = tf.nn.relu(coattention_context + b)
            coattention_context = tf.nn.dropout(coattention_context, self.dropout_placeholder)

        # Now process as in other baseline models

        projector_start = tf.get_variable(name="projectorS", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer())
        projector_end = tf.get_variable(name="projectorE", shape=(coattention_context.shape[2],), dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())

        prob_start = tf.einsum('ijk,k->ij', coattention_context, projector_start) * float_mask
        prob_end = tf.einsum('ijk,k->ij', coattention_context, projector_end) * float_mask

        logging.info("prob_start={}".format(prob_start))
        prob_start = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(prob_start)
        prob_start = prob_start * float_mask
        logging.info("prob_start={}".format(prob_start))

        prob_end_correlated = tf.concat([prob_end, tf.nn.softmax(prob_start)], axis=1)
        logging.info("prob_end_correlated={}".format(prob_end_correlated))
        # prob_end_correlated = tf.contrib.keras.layers.Dense(self.max_c_length, activation='relu')(prob_end_correlated)
        prob_end_correlated = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(prob_end_correlated)
        logging.info("prob_end_correlated={}".format(prob_end_correlated))

        prob_end_correlated = prob_end_correlated * float_mask

        masked_label_start = self.labels_placeholderS * int_mask
        masked_label_end = self.labels_placeholderE * int_mask

        cross_entropy_start = tf.nn.softmax_cross_entropy_with_logits(labels=masked_label_start, logits=prob_start,
                                                                      name="cross_entropy_start")
        cross_entropy_end = tf.nn.softmax_cross_entropy_with_logits(labels=masked_label_end, logits=prob_end_correlated,
                                                                    name="cross_entropy_end")
        prediction_start = tf.argmax(prob_start, 1)
        prediction_end = tf.argmax(prob_end_correlated, 1)

        logging.info("cross_entropy_end={}".format(cross_entropy_end))

        loss = tf.reduce_mean(cross_entropy_start) + tf.reduce_mean(cross_entropy_end)

        logging.info("loss.shape={}".format(loss.shape))
        return prediction_start, prediction_end, loss
