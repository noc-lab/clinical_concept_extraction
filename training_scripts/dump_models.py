import tensorflow as tf
from clinical_concept_extraction.model import annotation_ensemble

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('save_model_dir',
                    '../ckpt/bilstm_crf_concept/model_0/',
                    'path to the directory for saving the model')

flags.DEFINE_integer('random_seed', 0, 'random seed for training')


def main():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 100, 1024, 3])
        l = tf.placeholder(tf.int64, [None])

        y_ = annotation_ensemble(x, l)

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
