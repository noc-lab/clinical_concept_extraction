import tensorflow as tf
import re
from clinical_concept_extraction.model import bidirectional_lstm_func_freeze

flags = tf.app.flags
FLAGS = flags.FLAGS

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


def annotation_func_test(x, l, scope='clinical_concept_extraction'):
    with tf.variable_scope(scope):
        l = tf.cast(l, tf.int32)
        all_prediction = []

        for model_id in range(10):
            with tf.variable_scope('copy_' + str(model_id)):
                weight = tf.get_variable('weight', [3, 1], tf.float32, tf.constant_initializer(1))
                n_weight = tf.nn.softmax(weight, axis=0)
                gamma = tf.get_variable('gamma', [], tf.float32, tf.constant_initializer(1))
                token_embedding = tf.tensordot(x, n_weight, [[-1], [0]])
                token_embedding = gamma * tf.squeeze(token_embedding, axis=-1)

                lstm_output = bidirectional_lstm_func_freeze(token_embedding, l)

                logits = tf.layers.dense(lstm_output, 7, activation=None)

                transition = tf.get_variable('transitions', shape=[7, 7], dtype=tf.float32)

                viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(logits, transition, l)
                prediction = tf.cast(viterbi_sequence, tf.int32)
                all_prediction.append(prediction)

        all_prediction = tf.stack(all_prediction, axis=-1)

        return all_prediction,


def main():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 100, 1024, 3])
        l = tf.placeholder(tf.int64, [None])

        y_ = annotation_func_test(x, l)

        savers = []
        for model_id in range(10):
            scope = 'clinical_concept_extraction/copy_' + str(model_id)
            variables = tf.global_variables(scope)
            savers.append(tf.train.Saver(variables))

        final_saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            for model_id in range(10):
                savers[model_id].restore(
                    sess, '../ckpt/bilstm_crf_concept/model_' + str(model_id) + '/final_model')

            final_saver.save(sess, '../ckpt/ensemble/model', write_meta_graph=False)


if __name__ == '__main__':
    main()
