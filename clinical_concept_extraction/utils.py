import nltk
import re
from spacy import displacy
from simple_sentence_segment import sentence_segment



def parse_text(text):
    # Perform sentence segmentation, tokenization and return the lists of tokens,
    # spans, and text for every sentence respectively
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    all_sentences = []
    all_spans = []
    start = 0
    normalized_text = ''
    for span in sentence_segment(text):
        sentence = text[span[0]:span[1]]
        sentence = re.sub('\n', ' ', sentence)
        sentence = re.sub(r'\ +', ' ', sentence)
        sentence = sentence.strip()

        if len(sentence) > 0:
            tokens_span = tokenizer.span_tokenize(sentence)
            tokens = []
            spans = []
            for span in tokens_span:
                tokens.append(sentence[span[0]:span[1]])
                spans.append([start + span[0], start + span[1]])
                
            all_sentences.append(tokens)
            all_spans.append(spans)
            
            start += len(sentence) + 1
            normalized_text += sentence + '\n'
    return all_sentences, all_spans, normalized_text.strip()







def get_ents (listOfdic_entities , text):
    listOfent_label = []
    for dic in listOfdic_entities:
        l = dic["label"]
        en = text[dic["start"]:dic["end"]]
        listOfent_label.append([en , l])
    
    return listOfent_label


def build_display_elements(annotations):
    display_entities = []
    for i, annotation in enumerate(annotations):
        token, span, label = annotation
        if label !='O':
            new_ent = {}
            new_ent['start'] = span[0]
            new_ent['end'] = span[1]+1
            new_ent['label'] = label[2:]
            display_entities.append(new_ent)
    return display_entities
