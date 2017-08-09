import tensorflow as tf
import logging
from abstract_model import Qa_model


class Simple_baseline_qa_model(Qa_model):
    """The idea of this model is the following:
    1. Run question through a GRU RNN. Call the final state the question representation.
    2. Run context through the same GRU RNN and use the question representation as initial value.
    3. For each token in the context we obtain a vector from the above RNN. Project each of those vectors onto the 
    question representation to obtain a knowledge vector, which consists of one float for each token in the context.
    4. Feed the knowledge vector through two different softmaxes. This results in vectors xs and xe. Each float in xs 
    corresponds to a token in the context and represents the probability that this token is the start of the answer. 
    xe has a similar interpretation, just for the end of the answer.
    5. The loss is defined via cross entropy. 
    
    The point of this model is to check if the general setup in abstract_model has a bug. If this simple model would 
    not learn anything (exact_match less than 1%), there would be a bug somewhere."""

    def add_prediction_and_loss(self):
        WEM = tf.get_variable(name="WordEmbeddingMatrix", initializer=tf.constant(self.WordEmbeddingMatrix),
                              trainable=False)

        embedded_q = tf.nn.embedding_lookup(params=WEM, ids=self.q_input_placeholder)
        embedded_c = tf.nn.embedding_lookup(params=WEM, ids=self.c_input_placeholder)

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

        attention = tf.einsum('ik,ijk->ij', question_rep, c_outputs)
        float_mask = tf.cast(self.c_mask_placeholder, dtype=tf.float32)
        knowledge_vector = attention * float_mask

        xe = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(knowledge_vector)
        xs = tf.contrib.keras.layers.Dense(self.max_c_length, activation='linear')(knowledge_vector)

        xs = xs * float_mask
        xe = xe * float_mask

        cross_entropyS = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholderS, logits=xs,
                                                                 name="cross_entropyS")
        cross_entropyE = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholderE, logits=xe,
                                                                 name="cross_entropyE")
        predictionS = tf.argmax(xs, 1)
        predictionE = tf.argmax(xe, 1)

        logging.info("cross_entropyE.shape={}".format(cross_entropyE.shape))

        loss = tf.reduce_mean(cross_entropyS) + tf.reduce_mean(cross_entropyE)

        logging.info("loss.shape={}".format(loss.shape))
        return predictionS, predictionE, loss
