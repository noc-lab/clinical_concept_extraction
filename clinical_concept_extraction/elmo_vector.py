import numpy as np
import tensorflow as tf
from clinical_concept_extraction.bilm import Batcher, BidirectionalLanguageModel
import os

class ELMO_MIMIC:
    def __init__(self):
        base_path = os.path.join(os.environ['CCE_ASSETS'], 'elmo')
        if not os.path.isdir(base_path):
            raise FileNotFoundError

        vocab_file = os.path.join(base_path,'vocab.txt')
        options_file = os.path.join(base_path,'options.json')
        weight_file = os.path.join(base_path,'mimic_wiki.hdf5')

        self.batcher = Batcher(vocab_file, 50)
        self.input = tf.placeholder('int32', shape=(None, None, 50))
        self.model = BidirectionalLanguageModel(options_file, weight_file)
        self.output = self.model(self.input)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

    def get_embeddings(self, sentence):
        sentence_ids = self.batcher.batch_sentences([sentence])
        embedding = self.session.run(self.output['lm_embeddings'], feed_dict={self.input: sentence_ids})
        embedding = np.transpose(embedding[0], [1, 2, 0])

        return embedding

    def close_session(self):
        self.session.close()