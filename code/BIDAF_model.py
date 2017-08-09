import tensorflow as tf
import logging
from abstract_model import Qa_model

## WORK IN PROGRESS ##


class BIDAF_qa_model(Qa_model):
    """This is an implementation of the Dynamic Coattention Network model (https://arxiv.org/abs/1611.01604).
    It is work in progress. Right now, a simplified DCN encoder is implemented, which uses GRU instead of LSTMs and 
    doesn't use sentinel vectors yet. 
    Instead of a decoder from the above paper, there are two very simple baseline decoders implemented.
    """

    def add_prediction_and_loss(self):
        coattention_context = self.encode_bi_rnn()
        # prediction_start, prediction_end, loss = self.decode_with_baseline_decoder1(coattention_context)
        prediction_start, prediction_end, loss = self.decode_with_baseline_decoder2(coattention_context)

        return prediction_start, prediction_end, loss

    def encode_bi_rnn(self):
        self.WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                                   trainable=False)

        self.embedded_q = tf.nn.embedding_lookup(params=self.WEM, ids=self.q_input_placeholder)
        self.embedded_c = tf.nn.embedding_lookup(params=self.WEM, ids=self.c_input_placeholder)

        logging.info("embedded_q.shape={}".format(self.embedded_q.shape))
        logging.info("embedded_c.shape={}".format(self.embedded_c.shape))
        logging.info("labels_placeholderS.shape={}".format(self.labels_placeholderS.shape))

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

        logging.info("q_outputs_fw={}".format(q_outputs_fw))
        U = tf.concat([q_outputs_fw, q_outputs_bw], axis=2)
        U = tf.transpose(U,[0,2,1])

        logging.info("U={}".format(U))



        with tf.variable_scope("rnn", reuse=True):
            (c_outputs_fw, c_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                              inputs=self.embedded_c,
                                                                              sequence_length=c_sequence_length,
                                                                              dtype=tf.float32)

        H = tf.concat([c_outputs_fw, c_outputs_bw], axis=2)
        H = tf.transpose(H, [0, 2, 1])
        logging.info("H={}".format(H))

        S = tf.einsum('bij,bik->bjk',H,U)

        logging.info("S={}".format(S))

        assert False