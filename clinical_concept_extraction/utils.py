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




def build_display_elements(tokens, annotations, spans):
    # convert the annotations to the format used in displacy
    all_ann = []

    for sent_id, sent_info in enumerate(tokens):
        sent_length = len(tokens[sent_id])

        last_ann = 'O'
        last_start = None
        last_end = None
        for token_id in range(sent_length):
            this_ann = annotations[sent_id][token_id]

            # separated cases:
            if this_ann != last_ann:
                if last_ann != 'O':
                    # write last item
                    new_ent = {}
                    new_ent['start'] = last_start
                    new_ent['end'] = last_end
                    new_ent['label'] = last_ann[2:]
                    all_ann.append(new_ent)

                # record this instance
                last_ann = 'O' if this_ann == 'O' else 'I' + this_ann[1:]
                last_start = spans[sent_id][token_id][0]
                last_end = spans[sent_id][token_id][1]

            else:
                last_ann = this_ann
                last_end = spans[sent_id][token_id][1]

        if last_ann != 'O':
            new_ent = {}
            new_ent['start'] = last_start
            new_ent['end'] = last_end
            new_ent['label'] = last_ann[2:]
            all_ann.append(new_ent)

    return all_ann

