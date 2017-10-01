import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from simple_baseline_model import Simple_baseline_qa_model
from DCN_model import DCN_qa_model

tf.app.flags.DEFINE_string("model", "DCN", "Choose which model to use baseline/DCN")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 20, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("rnn_state_size", 64, "Size of RNNs used in the model.")
tf.app.flags.DEFINE_string("figure_directory", "figs/", "Directory in which figures are stored.")
tf.app.flags.DEFINE_float("dropout", 0.8, "1-Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_float("l2_lambda", 0.01, "Hyperparameter for l2 regularization.")
tf.app.flags.DEFINE_string("batch_permutation", "random",
                           "Choose whether training data is shuffled ('random'), ordered by length ('by_length'), "
                           "or kept in initial order ('None') for each epoch")
tf.app.flags.DEFINE_integer("decrease_lr", 0, "Whether to decrease lr over time")
tf.app.flags.DEFINE_float("lr_d_base", 0.9997, "Base for the exponential decay of lr")
tf.app.flags.DEFINE_float("lr_divider", 2, "Due to exp. decay, lr can go down to lr/lr_divider")
tf.app.flags.DEFINE_string("data_dir", "data/squad/", "SQuAD data directory")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # I do not use the code supplied by the CS224n staff for this main method
    # Just create a model, train it and evaluate on validation set

    if not os.path.exists(FLAGS.figure_directory):
        os.makedirs(FLAGS.figure_directory)

    if FLAGS.model == "baseline":
        model = Simple_baseline_qa_model(max_q_length=65, max_c_length=780, FLAGS=FLAGS)
    elif FLAGS.model == "DCN":
        # model = DCN_qa_model(max_q_length=65, max_c_length=780, FLAGS=FLAGS)
        model = DCN_qa_model(max_q_length=30, max_c_length=400, FLAGS=FLAGS)
    else:
        raise ValueError("model must be either 'baseline' or 'DCN'")
    model.train()


if __name__ == "__main__":
    tf.app.run()
