import numpy as np
import tensorflow as tf
from tqdm import trange
import logging
import matplotlib as mpl
from collections import Counter

mpl.use('Agg')  # can run on machine without display
import matplotlib.pyplot as plt
import os, re, string


class Qa_model(object):
    """
    The general framework of the question answering model is as follows (assuming batch size 1 for easier explanation):

        - the question or self.q_input_placeholder is a vector of ints like [45 189 1722 7]. Each int corresponds to a 
        word via a dictionary so [45 189 1722 7] could mean "Where is the cat". 
        self.c_input_placeholder is similar and contains a context like
        "On a lovely saturday evening the cat went into the green house. Soon thereafter, it started raining"

        - The model should predict an answer by using words from the context (in the above example "green house"). It 
        should give an answer span (10-11 in the above example, counting "On" as word zero), such that indexing the 
        context with that span would give the answer. We encode the answer span 10-11 into two ints 10,11 and these 
        into two one hot encodings:
            labels_placeholderS = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] # a 1 at the 10th position
            labels_placeholderE = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] # a 1 at the 11th position

        - Given the q_input_placeholder and c_input_placeholder, the model needs to predict two predictions.
        prediction_start and prediction_end that should ideally match labels_placeholderS and labels_placeholderE

    This class Qa_model has all functionalities of a QuestionAnswering Model, such as preprocessing of data, evaluation 
    metrics, batch processing, training loop etc., but it misses the heart of the model: the  add_prediction_and_loss() 
    method, which actually defines how to go from input X to label y. The method add_prediction_and_loss() must be 
    implemented in a derived class.
    """

    def __init__(self, max_q_length, max_c_length, FLAGS):
        self.max_q_length = max_q_length  # all questions will be cut or padded to have max_q_length
        self.max_c_length = max_c_length
        self.FLAGS = FLAGS
        logging.getLogger().setLevel(logging.INFO)

        self.load_and_preprocess_data()
        self.unit_tests()
        self.build_model()

    def add_prediction_and_loss(self):
        raise NotImplementedError("Each Model must re-implement this method.")

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
            line = line[:length]  # TODO: Add code to get rid of data for which len(line)>length.
            # Note: If the context for one line is not taken, question and answer_span should also not be taken and
            # vice versa for the question. Priority not too high, because having 0.1% garbage data is not too bad
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
            if start_ids[i] < max_length and end_ids[i] < max_length:
                labelS[start_ids[i]], labelE[end_ids[i]] = 1, 1  # one hot encoding
            E.append(labelE)
            S.append(labelS)
        return np.array(S), np.array(E)

    def load_and_preprocess_data(self):
        """Read in the Word embedding matrix as well as the question and context paragraphs and bring them into the 
        desired numerical shape."""

        logging.info("Data preparation. This can take some seconds...")
        # load vocab
        with open(self.FLAGS.data_dir + "vocab.dat", "r") as f:
            self.vocab = f.readlines()
        self.vocab = [x[:-1] for x in self.vocab]
        # load word embedding
        if self.FLAGS.word_vec_dim == 300:
            self.WordEmbeddingMatrix = np.load(self.FLAGS.data_dir + "glove.trimmed.300.npz")['glove']
        elif self.FLAGS.word_vec_dim == 100:
            self.WordEmbeddingMatrix = np.load(self.FLAGS.data_dir + "glove.trimmed.100.npz")['glove']
        else:
            raise ValueError("word_vec_dim can be either 100 or 300")
        logging.debug("WordEmbeddingMatrix.shape={}".format(self.WordEmbeddingMatrix.shape))
        null_wordvec_index = self.WordEmbeddingMatrix.shape[0]
        # append a zero vector to WordEmbeddingMatrix, which shall be used as padding value
        self.WordEmbeddingMatrix = np.vstack((self.WordEmbeddingMatrix, np.zeros(self.FLAGS.word_vec_dim)))
        self.WordEmbeddingMatrix = self.WordEmbeddingMatrix.astype(np.float32)
        logging.debug("WordEmbeddingMatrix.shape after appending zero vector={}".format(self.WordEmbeddingMatrix.shape))

        # load contexts, questions and labels
        self.yS, self.yE = self.span_to_y(np.loadtxt(self.FLAGS.data_dir + "train.span", dtype=np.int32))
        self.yvalS, self.yvalE = self.span_to_y(np.loadtxt(self.FLAGS.data_dir + "val.span", dtype=np.int32))

        self.X_c, self.X_c_mask = self.read_and_pad(self.FLAGS.data_dir + "train.ids.context", self.max_c_length,
                                                    null_wordvec_index)
        self.Xval_c, self.Xval_c_mask = self.read_and_pad(self.FLAGS.data_dir + "val.ids.context", self.max_c_length,
                                                          null_wordvec_index)
        self.X_q, self.X_q_mask = self.read_and_pad(self.FLAGS.data_dir + "train.ids.question", self.max_q_length,
                                                    null_wordvec_index)
        self.Xval_q, self.Xval_q_mask = self.read_and_pad(self.FLAGS.data_dir + "val.ids.question", self.max_q_length,
                                                          null_wordvec_index)

        logging.info("End data preparation.")

    ####################################################################################################################
    ######################## Model building ############################################################################
    ####################################################################################################################
    def build_model(self):
        self.add_placeholders()
        self.predictionS, self.predictionE, self.loss = self.add_prediction_and_loss()
        self.train_op, self.global_grad_norm = self.add_training_op(self.loss)

    def add_placeholders(self):
        self.q_input_placeholder = tf.placeholder(tf.int32, (None, self.max_q_length), name="q_input_ph")
        self.q_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_q_length),
                                                 name="q_mask_placeholder")
        self.c_input_placeholder = tf.placeholder(tf.int32, (None, self.max_c_length), name="c_input_ph")
        self.c_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, self.max_c_length),
                                                 name="c_mask_placeholder")
        self.labels_placeholderS = tf.placeholder(tf.int32, (None, self.max_c_length), name="label_phS")
        self.labels_placeholderE = tf.placeholder(tf.int32, (None, self.max_c_length), name="label_phE")

        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_ph")

    def add_training_op(self, loss):
        step_adam = tf.Variable(0, trainable=False)
        lr = tf.constant(self.FLAGS.learning_rate)
        if self.FLAGS.decrease_lr:
            # use adam optimizer with exponentially decaying learning rate
            rate_adam = tf.train.exponential_decay(lr, step_adam, 1, self.FLAGS.lr_d_base)
            # after one epoch: # 0.999**2500 = 0.5,  hence learning rate decays by a factor of 0.5 each epoch
            rate_adam = tf.maximum(rate_adam, tf.constant(self.FLAGS.learning_rate / self.FLAGS.lr_divider))
            # should not go down by more than a factor of 2
            optimizer = tf.train.AdamOptimizer(rate_adam)
        else:
            optimizer = tf.train.AdamOptimizer(lr)

        grads_and_vars = optimizer.compute_gradients(loss)
        variables = [output[1] for output in grads_and_vars]
        gradients = [output[0] for output in grads_and_vars]

        gradients = tf.clip_by_global_norm(gradients, clip_norm=self.FLAGS.max_gradient_norm)[0]
        global_grad_norm = tf.global_norm(gradients)
        grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=step_adam)
        self.optimizer = optimizer

        return train_op, global_grad_norm

    def get_feed_dict(self, batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE, keep_prob):
        feed_dict = {self.c_input_placeholder: batch_xc,
                     self.c_mask_placeholder: batch_xc_mask,
                     self.q_input_placeholder: batch_xq,
                     self.q_mask_placeholder: batch_xq_mask,
                     self.labels_placeholderS: batch_yS,
                     self.labels_placeholderE: batch_yE,
                     self.dropout_placeholder: keep_prob}
        return feed_dict

    ####################################################################################################################
    ######################## Evaluation metrics and plotting (of those metrics) ########################################
    ####################################################################################################################
    def squad_normalize_answer(self, s):
        """ Lower text and remove punctuation, articles and extra whitespace.
        Method copied from the SQuAD Leaderboard: https://rajpurkar.github.io/SQuAD-explorer/  """

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def squad_f1_score(self, prediction, ground_truth):
        """Method copied from the SQuAD Leaderboard: https://rajpurkar.github.io/SQuAD-explorer/"""
        prediction_tokens = self.squad_normalize_answer(prediction).split()
        ground_truth_tokens = self.squad_normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def squad_exact_match_score(self, prediction, ground_truth):
        """Method copied from the SQuAD Leaderboard: https://rajpurkar.github.io/SQuAD-explorer/"""
        return (self.squad_normalize_answer(prediction) == self.squad_normalize_answer(ground_truth))

    def get_f1(self, yS, yE, ypS, ypE):
        """My own, more strict f1 metric"""
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
            if n_true_pos != 0:
                precision = 1.0 * n_true_pos / n_pred_pos
                recall = 1.0 * n_true_pos / n_actual_pos
                f1_tot += (2 * precision * recall) / (precision + recall)
        f1_tot /= len(yS)
        return f1_tot

    def get_exact_match(self, yS, yE, ypS, ypE):
        """My own, more strict EM metric"""
        count = 0
        for i in range(len(yS)):
            s, e = np.argmax(yS[i]), np.argmax(yE[i])
            sp, ep = ypS[i], ypE[i]
            if sp > ep:
                sp, ep = ep, sp  # allow flipping between start and end
            if s == sp and e == ep:
                count += 1
        match_fraction = count / float(len(yS))
        return match_fraction

    def index_list_to_string(self, index_list):
        """Helper function. Converts a list of word indices to a string of words"""
        res = [self.vocab[index] for index in index_list]
        return ' '.join(res)

    def get_exact_match_from_tokens(self, yS, yE, ypS, ypE, batch_Xc):
        """This function doesn't compare the indices, but the tokens behind the indices. This is a bit more forgiving
        and it is the metric applied on the SQuAD leaderboard"""
        em = 0
        for i in range(len(yS)):
            s, e = np.argmax(yS[i]), np.argmax(yE[i])
            sp, ep = ypS[i], ypE[i]
            if sp > ep:
                sp, ep = ep, sp  # allow flipping between start and end
            ground_truth = self.index_list_to_string(batch_Xc[i][s:e + 1])
            prediction = self.index_list_to_string(batch_Xc[i][sp:ep + 1])
            em += self.squad_exact_match_score(prediction, ground_truth)
        return em / float(len(yS))

    def get_f1_from_tokens(self, yS, yE, ypS, ypE, batch_Xc):
        """This function doesn't compare the indices, but the tokens behind the indices. This is a bit more forgiving
        and it is the metric applied on the SQuAD leaderboard."""
        f1 = 0
        for i in range(len(yS)):
            s, e = np.argmax(yS[i]), np.argmax(yE[i])
            sp, ep = ypS[i], ypE[i]
            if sp > ep:
                sp, ep = ep, sp  # allow flipping between start and end
            ground_truth = self.index_list_to_string(batch_Xc[i][s:e + 1])
            prediction = self.index_list_to_string(batch_Xc[i][sp:ep + 1])
            f1 += self.squad_f1_score(prediction, ground_truth)
        return f1 / float(len(yS))

    def plot_metrics(self, prefix, index_epoch, losses, val_losses, EMs, val_Ems, F1s, val_F1s, grad_norms):
        n_data_points = len(losses)
        epoch_axis = np.arange(n_data_points, dtype=np.float32) * index_epoch / float(n_data_points)
        epoch_axis_val = list(range(index_epoch + 1))

        plt.plot(epoch_axis, losses, label="training")
        plt.plot(epoch_axis_val, [losses[0]] + val_losses, label="validation", marker="x", ms=15)
        # initial value is from training set, just so that we have a nice value for epoch=0 in the plot
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(self.FLAGS.figure_directory + prefix + "losses_over_time.png")
        plt.close()

        plt.plot(epoch_axis, EMs, label="training")
        plt.plot(epoch_axis_val, [EMs[0]] + val_Ems, label="validation", marker="x", ms=15)
        plt.xlabel("epoch")
        plt.ylabel("EM")
        plt.legend()
        plt.savefig(self.FLAGS.figure_directory + prefix + "EMs_over_time.png")
        plt.close()

        plt.plot(epoch_axis, F1s, label="training")
        plt.plot(epoch_axis_val, [F1s[0]] + val_F1s, label="validation", marker="x", ms=15)
        plt.xlabel("epoch")
        plt.ylabel("F1")
        plt.legend()
        plt.savefig(self.FLAGS.figure_directory + prefix + "f1s_over_time.png")
        plt.close()

        plt.plot(epoch_axis, grad_norms)
        plt.xlabel("epoch")
        plt.ylabel("gradient_norm")
        plt.savefig(self.FLAGS.figure_directory + prefix + "training_grad_norms_over_time.png")
        plt.close()

    ####################################################################################################################
    ######################## Batch processing ##########################################################################
    ####################################################################################################################
    def initialize_batch_processing(self, permutation='None', n_samples=None):
        self.batch_index = 0
        if n_samples is not None:
            self.max_batch_index = n_samples
        if permutation == 'by_length':
            # sum over True/False gives number of words in each sample
            length_of_each_context_paragraph = np.sum(self.X_c_mask, axis=1)
            # permutation of data is chosen, such that the algorithm sees short context_paragraphs first
            self.batch_permutation = np.argsort(length_of_each_context_paragraph)
        elif permutation == 'random':
            self.batch_permutation = np.random.permutation(self.max_batch_index)  # random initial permutation
        elif (permutation == 'None' or permutation is None):  # no permutation
            self.batch_permutation = np.arange(self.max_batch_index)  # initial permutation = identity
        else:
            raise ValueError("permutation must be 'by_length', 'random' or 'None'")

    def next_batch(self, batch_size, permutation_after_epoch='None', val=False):
        if self.batch_index >= self.max_batch_index:
            # we went through one epoch. reset batch_index and initialize batch_permutation
            self.initialize_batch_processing(permutation=permutation_after_epoch)

        start = self.batch_index
        end = self.batch_index + batch_size

        if not val:
            Xcres = self.X_c[self.batch_permutation[start:end]]
            Xcmaskres = self.X_c_mask[self.batch_permutation[start:end]]
            Xqres = self.X_q[self.batch_permutation[start:end]]
            Xqmaskres = self.X_q_mask[self.batch_permutation[start:end]]
            yresS = self.yS[self.batch_permutation[start:end]]
            yresE = self.yE[self.batch_permutation[start:end]]
        else:
            Xcres = self.Xval_c[self.batch_permutation[start:end]]
            Xcmaskres = self.Xval_c_mask[self.batch_permutation[start:end]]
            Xqres = self.Xval_q[self.batch_permutation[start:end]]
            Xqmaskres = self.Xval_q_mask[self.batch_permutation[start:end]]
            yresS = self.yvalS[self.batch_permutation[start:end]]
            yresE = self.yvalE[self.batch_permutation[start:end]]

        self.batch_index += batch_size
        return Xcres, Xcmaskres, Xqres, Xqmaskres, yresS, yresE

    ####################################################################################################################
    ######################## Unit tests ################################################################################
    ####################################################################################################################
    def unit_tests(self):
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
        logging.debug("span_to_y passed the test")

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
        logging.debug("read_and_pad passed the test")

        ################## test for get_exact_match and for get_f1 ##################
        yS = np.array([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
        yE = np.array([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
        ypS = np.array([2, 0, 0], dtype=np.int32)
        ypE = np.array([3, 0, 3])
        assert np.isclose(self.get_exact_match(yS, yE, ypS, ypE), 0.33, atol=0.01)
        logging.debug("get_exact_match passed the test")
        assert np.isclose(self.get_f1(yS, yE, ypS, ypE), (1 + 1 / 2. + 0.) / 3., atol=0.01)
        logging.debug("get_f1 passed the test")
        # Test the more foregiving metrics
        batch_Xc = [[71, 72, 73, 74, 75], [81, 82, 83, 84, 85], [55, 52, 53, 54, 55]]
        assert np.isclose(self.get_exact_match_from_tokens(yS, yE, ypS, ypE, batch_Xc), 0.33, atol=0.01)
        logging.debug("get_exact_match_from_tokens passed the test")
        assert np.isclose(self.get_f1_from_tokens(yS, yE, ypS, ypE, batch_Xc), (1 + 1 / 2. + 0.5 / 1.25) / 3.,
                          atol=0.01)
        logging.debug("get_f1_from_tokens passed the test")

    ####################################################################################################################
    ######################## Training ##################################################################################
    ####################################################################################################################
    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        epochs = self.FLAGS.epochs
        batch_size = self.FLAGS.batch_size
        n_samples = len(self.yS)

        global_losses, global_EMs, global_f1s, global_grad_norms = [], [], [], []  # global means "over several epochs"
        EMs_val, F1s_val, loss_val = [], [], []  # exact_match- and F1-metrics as well as loss on the validation data
        SQ_global_EMs, SQ_global_f1s, SQ_EMs_val, SQ_F1s_val = [], [], [], []  # corresponding squad metrics

        for index_epoch in range(1, epochs + 1):
            progbar = trange(int(n_samples / batch_size))
            losses, ems, f1s, grad_norms = [], [], [], []
            sq_ems, sq_f1s = [], []
            self.initialize_batch_processing(permutation=self.FLAGS.batch_permutation, n_samples=n_samples)

            ############### train for one epoch ###############
            for _ in progbar:
                batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE = self.next_batch(
                    batch_size=batch_size, permutation_after_epoch=self.FLAGS.batch_permutation)
                feed_dict = self.get_feed_dict(batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE,
                                               self.FLAGS.dropout)
                _, current_loss, predictionS, predictionE, grad_norm, curr_lr = sess.run(
                    [self.train_op, self.loss, self.predictionS, self.predictionE, self.global_grad_norm,
                     self.optimizer._lr],
                    feed_dict=feed_dict)
                ems.append(self.get_exact_match(batch_yS, batch_yE, predictionS, predictionE))
                f1s.append(self.get_f1(batch_yS, batch_yE, predictionS, predictionE))
                sq_ems.append(self.get_exact_match_from_tokens(batch_yS, batch_yE, predictionS, predictionE, batch_xc))
                sq_f1s.append(self.get_f1_from_tokens(batch_yS, batch_yE, predictionS, predictionE, batch_xc))
                losses.append(current_loss)
                grad_norms.append(grad_norm)

                if len(losses) >= 20:
                    progbar.set_postfix({'loss': np.mean(losses), 'EM': np.mean(ems), 'SQ_EM': np.mean(sq_ems), 'F1':
                        np.mean(f1s), 'SQ_F1': np.mean(sq_f1s), 'grad_norm': np.mean(grad_norms), 'lr': curr_lr})
                    global_losses.append(np.mean(losses))
                    global_EMs.append(np.mean(ems))
                    global_f1s.append(np.mean(f1s))
                    SQ_global_EMs.append(np.mean(sq_ems))
                    SQ_global_f1s.append(np.mean(sq_f1s))
                    global_grad_norms.append(np.mean(grad_norms))
                    losses, ems, f1s, grad_norms = [], [], [], []
                    sq_ems, sq_f1s = [], []

            ############### After an epoch: evaluate on validation set ###############
            logging.info("Epoch {} finished. Doing evaluation on validation set...".format(index_epoch))
            self.initialize_batch_processing(permutation=self.FLAGS.batch_permutation,
                                             n_samples=len(self.yvalE))
            val_batch_size = batch_size  # can be a multiple of batch_size, but be sure to not run out of memory
            losses, ems, f1s = [], [], []
            sq_ems, sq_f1s = [], []
            for _ in range(int(len(self.yvalE) / val_batch_size)):
                batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS, batch_yE = self.next_batch(
                    batch_size=val_batch_size, permutation_after_epoch=self.FLAGS.batch_permutation, val=True)
                feed_dict = self.get_feed_dict(batch_xc, batch_xc_mask, batch_xq, batch_xq_mask, batch_yS,
                                               batch_yE, keep_prob=1)
                current_loss, predictionS, predictionE = sess.run([self.loss, self.predictionS, self.predictionE],
                                                                  feed_dict=feed_dict)
                ems.append(self.get_exact_match(batch_yS, batch_yE, predictionS, predictionE))
                f1s.append(self.get_f1(batch_yS, batch_yE, predictionS, predictionE))
                sq_ems.append(self.get_exact_match_from_tokens(batch_yS, batch_yE, predictionS, predictionE, batch_xc))
                sq_f1s.append(self.get_f1_from_tokens(batch_yS, batch_yE, predictionS, predictionE, batch_xc))
                losses.append(current_loss)

            loss_on_validation, EM_val, F1_val = np.mean(losses), np.mean(ems), np.mean(f1s)
            SQ_F1_val, SQ_EM_val = np.mean(sq_f1s), np.mean(sq_ems)
            logging.info("loss_val={}".format(loss_on_validation))
            logging.info("EM_val={}".format(EM_val))
            logging.info("F1_val={}".format(F1_val))
            logging.info("SQ_EM_val={}".format(SQ_EM_val))
            logging.info("SQ_F1_val={}".format(SQ_F1_val))
            EMs_val.append(EM_val)
            F1s_val.append(F1_val)
            SQ_EMs_val.append(SQ_EM_val)
            SQ_F1s_val.append(SQ_F1_val)
            loss_val.append(loss_on_validation)

            ############### do some plotting ###############
            # if index_epoch > 1:
            self.plot_metrics("strict_", index_epoch, global_losses, loss_val, global_EMs, EMs_val, global_f1s, F1s_val,
                              global_grad_norms)

            self.plot_metrics("SQuAD_", index_epoch, global_losses, loss_val, SQ_global_EMs, SQ_EMs_val, SQ_global_f1s,
                              SQ_F1s_val, global_grad_norms)
