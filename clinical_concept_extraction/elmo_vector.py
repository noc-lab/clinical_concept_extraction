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
        self.input = tf.compat.v1.placeholder('int32', shape=(None, None, 50))
        self.model = BidirectionalLanguageModel(options_file, weight_file)
        self.output = self.model(self.input)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.compat.v1.Session(config=config)
        self.session.run(tf.compat.v1.global_variables_initializer())
        


    # get all sentences embeddings as a batch instead of getting each sentence embeddings alone
    def get_embeddings(self, all_sentences):
        # convert each sentence into list of chars ids.
        # applying padding for the whole batch of sentence by the max sentence length.
        sentences_ids, lengths_list = self.batcher.batch_sentences(all_sentences)
        embedding = tf.transpose(self.output['lm_embeddings'], perm=[0, 2, 3, 1])
        # embeddings shape = [batch_size, max sentence length, 1024, 3]
        embedding = self.session.run(embedding, feed_dict={self.input: sentences_ids})
        return embedding, lengths_list


    def close_session(self):
        self.session.close()