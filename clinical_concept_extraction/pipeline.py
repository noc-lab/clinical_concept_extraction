import tensorflow as tf
import os
from clinical_concept_extraction.model import annotation_ensemble
from clinical_concept_extraction.elmo_vector import ELMO_MIMIC
from scipy.stats import mode

all_concept = ['O', 'I-problem', 'I-treatment', 'I-test', 'B-problem', 'B-treatment', 'B-test']


def build_inputs(all_sentences):
    all_x = []
    all_l = []

    elmo_models = ELMO_MIMIC()

    for sentence in all_sentences:
        embeddings = elmo_models.get_embeddings(sentence)
        all_x.append(embeddings)
        all_l.append(len(embeddings))
    elmo_models.close_session()

    return all_x, all_l


def build_generator(x, l):
    def generator():
        for x_, l_ in zip(x, l):
            yield x_, l_

    return generator


def get_annotation(all_sentences, batch_size=2):
    model_path = os.path.join(os.environ['CCE_ASSETS'], 'blstm')
    if not os.path.isdir(model_path):
        raise FileNotFoundError

    x, l = build_inputs(all_sentences)

    tf.reset_default_graph()
    with tf.Graph().as_default():
        dataset = tf.data.Dataset().from_generator(
            build_generator(x, l),
            (tf.float32, tf.int64),
            (tf.TensorShape([None, 1024, 3]), tf.TensorShape([]))
        )

        dataset = dataset.padded_batch(batch_size, ([tf.Dimension(None), tf.Dimension(1024), tf.Dimension(3)], []))

        iterator = dataset.make_initializable_iterator()
        x_, l_ = iterator.get_next()

        y = annotation_ensemble(x_, l_)

        saver = tf.train.Saver()
        all_y = []

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, os.path.join(model_path, 'model'))
            sess.run(iterator.initializer)
            while True:
                try:
                    all_y += list(sess.run([y])[0][0])
                except tf.errors.OutOfRangeError:
                    break

    all_y_ens = []
    for i in range(len(l)):
        best_v, _ = mode(all_y[i][:l[i]], axis=1)
        ann_ids = best_v.reshape(-1)
        ann = [all_concept[i] for i in ann_ids]
        all_y_ens.append(ann)

    return all_y_ens
