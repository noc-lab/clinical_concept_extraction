import tensorflow as tf
import os
from clinical_concept_extraction.model import build_clinical_graph
from clinical_concept_extraction.elmo_vector import ELMO_MIMIC
from simple_sentence_segment import sentence_segment
from scipy.stats import mode
import numpy as np
from clinical_concept_extraction.utils import parse_text
import time

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

class ClinicalConceptExtraction:
    def __init__(self, models_path):
        if not os.path.isdir(models_path):
            raise FileNotFoundError
        os.environ['CCE_ASSETS'] = models_path
        
        self.all_concept = ['O', 'I-problem', 'I-treatment', 'I-test', 'B-problem', 'B-treatment', 'B-test']

        # build both emlo and clinical_concept extraction globally so no need to build them with each time prediction needed
        # save some time and speed up prediction process.
        self.elmo_model = ELMO_MIMIC()
        tf.compat.v1.reset_default_graph()
        self.y, self.x_placeHolder, self.l_placeHolder, self.clinical_session = build_clinical_graph(session=tf.compat.v1.Session(config=config))


    def decode_prediction(self, all_y, l):
        '''
        map prediction output to all concepts
        ['O', 'I-problem', 'I-treatment', 'I-test', 'B-problem', 'B-treatment', 'B-test']
        '''
        all_y_ens = []
        for i in range(len(l)):
            best_v, _ = mode(all_y[i][:l[i]], axis=1)
            ann_ids = best_v.reshape(-1)
            ann = [self.all_concept[i] for i in ann_ids]
            all_y_ens.append(ann)

        return all_y_ens


    def predict_concepts_labels(self,tokenized_sentences):
        '''
        get embeddings for batch tokenized sentences and feed them to the clinical concept extraction model.
        '''
        embedds, embedds_lengths = self.elmo_model.get_embeddings(tokenized_sentences)
        all_y = [self.clinical_session.run([self.y], feed_dict={self.x_placeHolder:embedds,self.l_placeHolder:embedds_lengths})[0][0]]
        prediction = self.decode_prediction(np.squeeze(all_y, axis=0), embedds_lengths)
        return prediction

    
    def extract_concepts(self, text, batch_size=1, as_one_batch=False):
        '''
        note: sample text
        as_one_batch : boolen to indicate if desired to predict the whole text as one batch
        '''
        start_time = time.time()
        concepts = []
        tokenized_sentences, all_spans, normalized_text = parse_text(text)
        
        if(batch_size> len(tokenized_sentences)) or as_one_batch:
            batch_size = len(tokenized_sentences)

        number_of_batches = int(len(tokenized_sentences)/batch_size)
        remaining_batchs = len(tokenized_sentences)%batch_size



        for batch_number in range(number_of_batches):
            batch_sentences_tokens = tokenized_sentences[batch_number*batch_size:(batch_number*batch_size)+batch_size]
            
            batch_spans = all_spans[batch_number*batch_size:(batch_number*batch_size)+batch_size]

            predictions=self.predict_concepts_labels(batch_sentences_tokens)
                    
            for sent_tokens, sent_spans, sent_ann in zip(batch_sentences_tokens, batch_spans, predictions):
                for token, span, annotation in zip(sent_tokens, sent_spans, sent_ann):
                    concepts.append([token,span, annotation])

        # predict remaining last batch
        if remaining_batchs > 0:
            remaining_last_batch = tokenized_sentences[number_of_batches*batch_size:]
            remaining_last_spans = all_spans[number_of_batches*batch_size:]
            predictions = self.predict_concepts_labels(remaining_last_batch)
            for sent_tokens, sent_spans, sent_ann in zip(remaining_last_batch, remaining_last_spans, predictions):
                for token, span, annotation in zip(sent_tokens, sent_spans, sent_ann):
                    concepts.append([token,span, annotation])



        print("\n\nTook ", time.time()-start_time, " Seconds to predict\n\n")


        # concept is an list of [[token_0, span_0, label_0], [token_1, span_1, label_1], ..., ...., [token_n, span_n, label_n]]
        return concepts







