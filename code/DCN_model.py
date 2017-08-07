import tensorflow as tf
import logging
from abstract_model import Qa_model


class DCN_qa_model(Qa_model):
    def add_prediction_and_loss(self):
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
            question_rep = q_final_state

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
            u_outputs, u_state = tf.nn.dynamic_rnn(cell, inputs=CDprime, dtype=tf.float32,
                                                   sequence_length=c_sequence_length)

        logging.info("u_outputs.shape={}".format(u_outputs.shape))

        ############### simple decoding with rnn ###############

        projector = tf.get_variable(name="projector",shape=(2*rnn_size,),dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

        knowledge_vector = tf.einsum('ijk,k->ij',u_outputs,projector)
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

        cross_entropyS = tf.nn.softmax_cross_entropy_with_logits(labels=mls, logits=xs,
                                                                 name="cross_entropyS")
        cross_entropyE = tf.nn.softmax_cross_entropy_with_logits(labels=mle, logits=xe,
                                                                 name="cross_entropyE")
        predictionS = tf.argmax(xs, 1)
        predictionE = tf.argmax(xe, 1)

        logging.info("cross_entropyE.shape={}".format(cross_entropyE.shape))

        loss = tf.reduce_mean(cross_entropyS) + tf.reduce_mean(cross_entropyE)

        logging.info("loss.shape={}".format(loss.shape))
        return predictionS, predictionE, loss

