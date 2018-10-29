import os
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import pickle
from clinical_concept_extraction.model import bidirectional_rnn_func

flags = tf.app.flags
FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_string('save_model_dir',
                    '../ckpt/bilstm_crf_concept/model_0/',
                    'path to the directory for saving the model')

flags.DEFINE_string('tfrecord_dir',
                    '../data/preprocessed/tfrecords/',
                    'path to the directory for tfrecords')

flags.DEFINE_string('eval_dir',
                    '../evaluate/pkl/',
                    'path to the directory for evaluation')

flags.DEFINE_integer('random_seed', 0, 'random seed for training')

flags.DEFINE_integer('batch_size', 32, 'Batch size of the model')
flags.DEFINE_integer('num_epochs', 200, 'Number of the epochs in training')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate of the optimizer')

flags.DEFINE_bool('train', False, 'train or eval')


def annotation_func_train(x, y, l, train, reuse, scope='clinical_concept_extraction'):
    with tf.variable_scope(scope, reuse=reuse):
        l = tf.cast(l, tf.int32)
        # find logit
        with tf.variable_scope('copy_' + str(FLAGS.random_seed)):
            weight = tf.get_variable('weight', [3, 1], tf.float32, tf.constant_initializer(1))
            n_weight = tf.nn.softmax(weight, axis=0)
            gamma = tf.get_variable('gamma', [], tf.float32, tf.constant_initializer(1))
            token_embedding = tf.tensordot(x, n_weight, [[-1], [0]])
            token_embedding = gamma * tf.squeeze(token_embedding, axis=-1)

            lstm_output = bidirectional_rnn_func(token_embedding, l, train)

            logits = tf.layers.dense(lstm_output, 7, activation=None)

            log_likelihood, transition_params = \
                tf.contrib.crf.crf_log_likelihood(logits, y, l)
            viterbi_sequence, viterbi_score = \
                tf.contrib.crf.crf_decode(logits, transition_params, l)

        prediction = tf.cast(viterbi_sequence, tf.int32)

        step_loss = tf.reduce_mean(-log_likelihood)

        if train:
            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(step_loss)
            return step_loss, transition_params, train_step, prediction
        else:
            return step_loss, transition_params, prediction


def annotation_func_test(x, l, reuse, scope='clinical_concept_extraction'):
    with tf.variable_scope(scope, reuse=reuse):
        l = tf.cast(l, tf.int32)
        # project to some low dimension
        # find logit
        with tf.variable_scope('copy_' + str(FLAGS.random_seed)):
            weight = tf.get_variable('weight', [3, 1], tf.float32, tf.constant_initializer(1))
            n_weight = tf.nn.softmax(weight, axis=0)
            gamma = tf.get_variable('gamma', [], tf.float32, tf.constant_initializer(1))
            token_embedding = tf.tensordot(x, n_weight, [[-1], [0]])
            token_embedding = gamma * tf.squeeze(token_embedding, axis=-1)

            lstm_output = bidirectional_rnn_func(token_embedding, l, False)

            logits = tf.layers.dense(lstm_output, 7, activation=None)

            transition = tf.get_variable('transitions', shape=[7, 7], dtype=tf.float32)

            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(logits, transition, l)
        prediction = tf.cast(viterbi_sequence, tf.int32)

        return prediction,


def _parse_function(example_proto):
    contexts, features = tf.parse_single_sequence_example(
        example_proto,
        context_features={
            "length": tf.FixedLenFeature([], dtype=tf.int64),
        },
        sequence_features={
            "token": tf.FixedLenSequenceFeature([1024 * 3], dtype=tf.float32),
            "concept": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
    )

    token = tf.reshape(features["token"], (-1, 1024, 3))

    return token, features["concept"], contexts["length"]


def find_marco_f1(all_y_pred, all_y, all_l):
    """
    Note this is NOT the official evaluation script.
    This is a estimation for token-level marco f1 during training
    """
    tag_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}
    confusion_matrix = np.zeros((4, 4), dtype=int)
    for y_pred_ins, y_ins, l_ins in zip(all_y_pred, all_y, all_l):
        for i in range(1, l_ins):
            confusion_matrix[tag_map[y_ins[i]], tag_map[y_pred_ins[i]]] += 1

    small_conf_mat = np.zeros([2, 2])
    for class_id in range(1, 4):
        if np.sum(confusion_matrix[:, class_id]) > 0:
            non_class_id = [i for i in range(4) if i != class_id]
            small_conf_mat[0, 0] += confusion_matrix[class_id, class_id]
            small_conf_mat[1, 0] += np.sum(confusion_matrix[non_class_id, class_id])
            small_conf_mat[0, 1] += np.sum(confusion_matrix[class_id, non_class_id])
            small_conf_mat[1, 1] += np.sum(confusion_matrix[non_class_id, non_class_id])

    try:
        micro_r = small_conf_mat[0, 0] / (np.sum(small_conf_mat[0, :]))
    except:
        micro_r = np.nan

    try:
        micro_p = small_conf_mat[0, 0] / (np.sum(small_conf_mat[:, 0]))
    except:
        micro_p = np.nan

    try:
        micro_f = 2 * (micro_p * micro_r) / (micro_p + micro_r)
    except:
        micro_f = np.nan

    return micro_p, micro_r, micro_f


def generate_iterator_ops(filenames, train=True, reuse=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)

    if train:
        dataset = dataset.shuffle(buffer_size=2 * FLAGS.batch_size)

    dataset = dataset.padded_batch(
        FLAGS.batch_size,
        ([tf.Dimension(None), tf.Dimension(1024), tf.Dimension(3)],
         [tf.Dimension(None)], [])
    )
    data_iterator = dataset.make_initializable_iterator()
    next_x, next_y, next_l = data_iterator.get_next()

    if train:
        ops = annotation_func_train(next_x, next_y, next_l, train=train, reuse=reuse)
    else:
        ops = annotation_func_test(next_x, next_l, reuse=reuse)

    ops = list(ops)
    ops.append(next_y)
    ops.append(next_l)

    return data_iterator, ops


def run_one_epoch_train(sess, iterator, ops):
    """Proceed a epoch of training/validation"""
    all_loss = []
    all_y_pred = []
    all_y = []
    all_l = []
    sess.run(iterator.initializer)

    while True:
        try:
            results = sess.run(ops)
            all_loss.append(results[0])

            all_y_pred += list(results[-3])
            all_y += list(results[-2])
            all_l += list(results[-1])
        except tf.errors.OutOfRangeError:
            break

    f1 = find_marco_f1(all_y_pred, all_y, all_l)

    return np.mean(all_loss), f1[2]


def run_one_epoch_test(sess, iterator, ops):
    """Proceed a epoch of training/validation"""
    all_y_pred = []
    all_y = []
    all_l = []
    sess.run(iterator.initializer)

    while True:
        try:
            results = sess.run(ops)
            all_y_pred += list(results[-3])
            all_y += list(results[-2])
            all_l += list(results[-1])
        except tf.errors.OutOfRangeError:
            break

    f1 = find_marco_f1(all_y_pred, all_y, all_l)

    return f1, all_y_pred, all_y, all_l


def train():
    all_files = os.listdir(FLAGS.tfrecord_dir)
    all_files = [item for item in all_files if item[:2] == 'tr']

    train_files = [FLAGS.tfrecord_dir + item for item in all_files]

    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.random_seed)

        train_iter, train_op = generate_iterator_ops(
            train_files, train=True, reuse=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        tf.gfile.MakeDirs(FLAGS.save_model_dir)

        table = []

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(FLAGS.num_epochs):
                tic = time.time()
                train_loss, f1_estimate = run_one_epoch_train(
                    sess, train_iter, train_op)
                toc = time.time() - tic

                print("Epoch %d: train loss %.4f, F1 %.4f, elapsed time %.1f s" %
                      (epoch, train_loss, f1_estimate, toc))

                table.append([epoch, train_loss, f1_estimate])

                saver.save(sess, FLAGS.save_model_dir + 'final_model',
                           write_state=False, write_meta_graph=False)

            table = pd.DataFrame(table)
            table.columns = ['epoch', 'loss:train', 'f1:train']
            table.to_csv(FLAGS.save_model_dir + 'progress.csv', index=False)


def evaluate():
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    tf.reset_default_graph()
    with tf.Graph().as_default():
        valid_iter, valid_op = generate_iterator_ops(
            [FLAGS.tfrecord_dir + 'test.tfrecords'],
            train=False, reuse=False
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.save_model_dir + 'final_model')

            f1_estimate, logits, all_y, all_l = run_one_epoch_test(sess, valid_iter, valid_op)

            print(f1_estimate)
    pickle.dump([logits, all_l], open(FLAGS.eval_dir + str(FLAGS.random_seed) + '.pkl', 'wb'))


def main(_):
    if FLAGS.train:
        train()
    else:
        evaluate()


if __name__ == '__main__':
    tf.app.run()
