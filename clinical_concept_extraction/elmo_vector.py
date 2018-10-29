import numpy as np
import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel


class ELMO_MIMIC:
    def __init__(self):
        base_path = '/home/henghuiz/HugeData/datasets/processed/elmo_clinical_open/'

        vocab_file = base_path + 'vocab.txt'
        options_file = base_path + 'options.json'
        weight_file = base_path + 'mimic_wiki.hdf5'

        self.batcher = Batcher(vocab_file, 50)
        self.input = tf.placeholder('int32', shape=(None, None, 50))
        self.model = BidirectionalLanguageModel(options_file, weight_file)
        self.output = self.model(self.input)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def get_embeddings(self, sentence):
        sentence_ids = self.batcher.batch_sentences([sentence])
        embedding = self.session.run(self.output['lm_embeddings'], feed_dict={self.input: sentence_ids})
        embedding = np.transpose(embedding[0], [1, 2, 0])

        return embedding
