import tensorflow as tf
import logging
from abstract_model import Qa_model


class Simple_baseline_qa_model(Qa_model):
    def add_prediction_and_loss(self):
        WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                              trainable=False)

        embedded_q = tf.nn.embedding_lookup(params=WEM, ids=self.q_input_placeholder)
        embedded_c = tf.nn.embedding_lookup(params=WEM, ids=self.c_input_placeholder)

        logging.info("embedded_q.shape={}".format(embedded_q.shape))
        logging.info("embedded_c.shape={}".format(embedded_c.shape))
        logging.info("labels_placeholderS.shape={}".format(self.labels_placeholderS.shape))

        rnn_size = self.FLAGS.rnn_state_size
        with tf.variable_scope("rnn", reuse=None):
            cell = tf.contrib.rnn.GRUCell(rnn_size)
            q_sequence_length = tf.reduce_sum(tf.cast(self.q_mask_placeholder, tf.int32), axis=1)
            q_sequence_length = tf.reshape(q_sequence_length, [-1, ])
            c_sequence_length = tf.reduce_sum(tf.cast(self.c_mask_placeholder, tf.int32), axis=1)
            c_sequence_length = tf.reshape(c_sequence_length, [-1, ])

            q_outputs, q_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=embedded_q,
                                                         sequence_length=q_sequence_length, dtype=tf.float32,
                                                         time_major=False)
            question_rep = q_final_state

        with tf.variable_scope("rnn", reuse=True):
            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=embedded_c,
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
        predictionS = tf.argmax(xs, 1)
        predictionE = tf.argmax(xe, 1)

        logging.info("cross_entropyE.shape={}".format(cross_entropyE.shape))

        loss = tf.reduce_mean(cross_entropyS) + tf.reduce_mean(cross_entropyE)

        logging.info("loss.shape={}".format(loss.shape))
        return predictionS, predictionE, loss
