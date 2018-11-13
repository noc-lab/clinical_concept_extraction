import os
import tensorflow as tf
import pickle
from sklearn.model_selection import KFold
from tqdm import tqdm

from clinical_concept_extraction.elmo_vector import ELMO_MIMIC
from html import unescape


def clean_text(t, c):
    """
    There will be some empty token that should be clean in the first place
    """
    new_t = []
    new_c = []

    for t_, c_ in zip(t, c):
        if len(t_) > 0:
            new_t.append(unescape(t_))
            new_c.append(c_)

    return new_t, new_c


def write_tf_records(list_t, list_c, output_filename, all_c, elmo_model):
    concept2id = {concept: concept_id for concept_id, concept in enumerate(all_c)}

    writer = tf.python_io.TFRecordWriter(output_filename)

    sent_id = 0

    for t, c in tqdm(zip(list_t, list_c), total=len(list_t)):
        t, c = clean_text(t, c)
        c = [concept2id[item] for item in c]
        l = len(c)

        embeddings = elmo_model.get_embeddings(t[:l])

        context = tf.train.Features(feature={  # Non-serial data uses Feature
            "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[l])),
        })

        token_features = [tf.train.Feature(float_list=tf.train.FloatList(value=embedding.reshape(-1))) for embedding
                          in
                          embeddings]
        concept_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[c_])) for c_ in c]

        feature_list = {
            'token': tf.train.FeatureList(feature=token_features),
            'concept': tf.train.FeatureList(feature=concept_features),
        }

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        ex = tf.train.SequenceExample(feature_lists=feature_lists, context=context)
        writer.write(ex.SerializeToString())

        sent_id += 1
    writer.close()


def main():
    all_concept = ['', 'problem', 'treatment', 'test', 'B-problem', 'B-treatment', 'B-test']

    save_dir = '../data/preprocessed/tfrecords/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    beth_t, beth_c = pickle.load(open('../data/preprocessed/pkl/beth.pkl', 'rb'))
    partners_t, partners_c = pickle.load(open('../data/preprocessed/pkl/partners.pkl', 'rb'))
    test_t, test_c = pickle.load(open('../data/preprocessed/pkl/text.pkl', 'rb'))

    train_t = beth_t + partners_t
    train_c = beth_c + partners_c

    elmo_model = ELMO_MIMIC()

    # not for cv, just to break to 10 shards
    cv = KFold(n_splits=10, random_state=0, shuffle=True)

    split_num = 0

    for _, valid_set in cv.split(train_c):
        valid_t = [train_t[i] for i in valid_set]
        valid_c = [train_c[i] for i in valid_set]

        output_filename = save_dir + 'train_cv' + str(split_num) + '.tfrecords'

        write_tf_records(valid_t, valid_c, output_filename, all_concept, elmo_model)

        split_num += 1

    output_filename = save_dir + 'test.tfrecords'

    write_tf_records(test_t, test_c, output_filename, all_concept, elmo_model)


if __name__ == '__main__':
    main()
