import numpy as np
import tensorflow as tf
from clinical_concept_extraction.bilm import Batcher, BidirectionalLanguageModel
import os
from tqdm import tqdm

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

        # self.vocab_embeddings()



    def get_embeddings(self, all_sentences):
        sentences_ids, lengths_list = self.batcher.batch_sentences(all_sentences)
        embedding = tf.transpose(self.output['lm_embeddings'], perm=[0, 2, 3, 1])
        embedding = self.session.run(embedding, feed_dict={self.input: sentences_ids})

        return embedding, lengths_list

    def vocab_embeddings(self):
        self.vocab_vocab_embeddings_list = []
        print("\n\n\n\n===========================================================")
        print("Vocab length = ",len(self.batcher.words_vocab))
        for word in tqdm(self.batcher.words_vocab):
            # print(i, " word > ", word)
            self.vocab_vocab_embeddings_list.append(self.get_embeddings(word))


    @property
    def get_vocab_embeddings(self):
        return self.vocab_vocab_embeddings_list

    def close_session(self):
        self.session.close()