import tensorflow as tf
import os
from clinical_concept_extraction.model import build_clinical_graph
from clinical_concept_extraction.elmo_vector import ELMO_MIMIC
from simple_sentence_segment import sentence_segment
from scipy.stats import mode
import numpy as np
from clinical_concept_extraction.utils import parse_text
import time
all_concept = ['O', 'I-problem', 'I-treatment', 'I-test', 'B-problem', 'B-treatment', 'B-test']
os.environ['CCE_ASSETS'] = '/home/omar/Desktop/kamil_clinic/cce_assets'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
elmo_model = ELMO_MIMIC()
tf.reset_default_graph()
y, x_placeHolder, l_placeHolder, clinical_session = build_clinical_graph(session=tf.Session(config=config))



def decode_prediction(all_y, l):
    all_y_ens = []
    for i in range(len(l)):
        best_v, _ = mode(all_y[i][:l[i]], axis=1)
        ann_ids = best_v.reshape(-1)
        ann = [all_concept[i] for i in ann_ids]
        all_y_ens.append(ann)

    return all_y_ens


def predict_concepts_labels(tokenized_sentences):
    embedds, embedds_lengths = elmo_model.get_embeddings(tokenized_sentences)
    all_y = [clinical_session.run([y], feed_dict={x_placeHolder:embedds,l_placeHolder:embedds_lengths})[0][0]]
    prediction = decode_prediction(np.squeeze(all_y, axis=0), embedds_lengths)
    return prediction

def extract_concepts(Note, batch_size=1):
    start_time = time.time()
    global elmo_model, clinical_session
    concepts = []
    tokenized_sentences, all_spans, normalized_text = parse_text(Note)
    
    if(batch_size> len(tokenized_sentences)):
        batch_size = len(tokenized_sentences)

    number_of_batches = int(len(tokenized_sentences)/batch_size)
    remaining_batchs = len(tokenized_sentences)%batch_size



    for batch_number in range(number_of_batches):
        batch_sentences_tokens = tokenized_sentences[batch_number*batch_size:(batch_number*batch_size)+batch_size]
        predictions=predict_concepts_labels(batch_sentences_tokens)
        for sent_, ann_ in zip(batch_sentences_tokens, predictions):
            for token, annotation in zip(sent_, ann_):
                concepts.append([token, annotation])

    # predict remaining last batch
    if remaining_batchs > 0:
        remaining_last_batch = tokenized_sentences[number_of_batches*batch_size:]
        predictions = predict_concepts_labels(remaining_last_batch)
        for sent_, ann_ in zip(remaining_last_batch, predictions):
            for token, annotation in zip(sent_, ann_):
                concepts.append([token, annotation])

    
    
    print("\n\nTook ", time.time()-start_time, " Seconds to predict\n\n")



    return concepts






# def build_inputs(all_sentences):
#     all_x = []
#     all_l = []

#     # elmo_models = ELMO_MIMIC()

#     for sentence in all_sentences:
#         embeddings = elmo_models.get_embeddings(sentence)
#         all_x.append(embeddings)
#         all_l.append(len(embeddings))


#     batch_size = len(all_l)
#     dataset = tf.data.Dataset.from_generator(lambda: zip(all_x, all_l), (tf.float32, tf.int64),(tf.TensorShape([None, 1024, 3]), tf.TensorShape([])))
#     dataset = dataset.padded_batch(batch_size, ([tf.Dimension(None), tf.Dimension(1024), tf.Dimension(3)], []))
#     iterator = dataset.make_one_shot_iterator()
#     next_element = iterator.get_next()
    
#     with tf.Session() as sess:
#         input_data=sess.run(next_element)

#     x_inp, l_inp = input_data

#     return x_inp, l_inp, all_l

# def build_generator(x, l):
#     def generator():
#         for x_, l_ in zip(x, l):
#             yield x_, l_

#     return generator






