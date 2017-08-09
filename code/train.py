# from __future__ import absolute_import
# from __future__ import division

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from simple_baseline_model import Simple_baseline_qa_model
from DCN_model import DCN_qa_model
from BIDAF_model import BIDAF_qa_model

import logging

logging.basicConfig(level=logging.DEBUG)
tf.app.flags.DEFINE_string("model", "baseline", "Choose which model to use baseline/DCN/BIDAF")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 20, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("rnn_state_size", 64, "Size of RNNs used in the model.")
tf.app.flags.DEFINE_string("figure_directory", "figs/", "Directory in which figures are stored.")
tf.app.flags.DEFINE_float("dropout", 0.9, "1-Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_string("batch_permutation", "random",
                           "Choose whether training data is shuffled ('random'), ordered by length ('by_length'), "
                           "or kept in initial order ('None') for each epoch")

tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad/", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "",
                           "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "",
                           "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # I do not use the code supplied by the CS224n staff for this main method
    # Just create a model, train it and evaluate on validation set
    # print FLAGS.learning_rate

    if not os.path.exists(FLAGS.figure_directory):
        os.makedirs(FLAGS.figure_directory)

    if FLAGS.model == "baseline":
        model = Simple_baseline_qa_model(max_q_length=65, max_c_length=780, FLAGS=FLAGS)
    elif FLAGS.model == "DCN":
        model = DCN_qa_model(max_q_length=65, max_c_length=780, FLAGS=FLAGS)
    elif FLAGS.model == "BIDAF":
        model = BIDAF_qa_model(max_q_length=65, max_c_length=780, FLAGS=FLAGS)
    else:
        raise ValueError("model must be either 'baseline' or 'DCN'")
    model.train()


if __name__ == "__main__":
    tf.app.run()
