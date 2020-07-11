from simple_sentence_segment import sentence_segment


class SentenceSegmenter(object):
    def __init__(self):
        pass

    def set_sent_starts(self, doc):
        for d in doc:
            d.is_sent_start = False
        for s, _ in sentence_segment(str(doc)):
            print(s)
            for token in doc:
                if token.idx >= s:
                    token.is_sent_start = True
                    break
        return doc
