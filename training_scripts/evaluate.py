import os
import pickle
import re
import shutil
import argparse
import tempfile
import subprocess

from scipy.stats import mode
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument("--pickle_file", type=str, default='../evaluate/pkl/0.pkl')
parser.add_argument("--save_path", type=str, default='../evaluate/pkl/')

parser.add_argument("--ensemble", type=bool, default=True)
parser.add_argument("--full_report", type=bool, default=False)

args = parser.parse_args()


def space_tokenizer(text):
    tokens = re.split('\ +', text)
    spans = []
    start = 0
    for token in tokens:
        if token == '':
            spans.append((start, start))
        else:
            this_span = re.search(re.escape(token), text[start:])
            assert this_span is not None
            this_span = this_span.span()
            spans.append((this_span[0] + start, this_span[1] + start))
            start += this_span[1]

    return tokens, spans


def convert_i2b2_format(all_y_pred, all_l):
    test_text_path = '../data/raw/test/txt/'
    all_concept = ['', 'problem', 'treatment', 'test']

    tag_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}

    output_path = tempfile.mkdtemp()
    output_path += '/'  # add slash to avoid a bug for jar

    all_txt_files = os.listdir(test_text_path)
    all_txt_files = [item for item in all_txt_files if item[-3:] == 'txt']
    all_txt_files.sort()

    for txt_filename in all_txt_files:
        text = open(test_text_path + txt_filename, 'r', encoding='utf-8-sig').read()

        all_sentences = text.split('\n')

        token_list = [space_tokenizer(sentence) for sentence in all_sentences]
        useful_sentences = [all_sentences[i] for i in range(len(all_sentences)) if len(token_list[i][0]) > 0]
        token_list = [sentence for sentence in token_list if len(sentence[0]) > 0]

        sent_id = 0
        all_annotation = []

        for sentence in token_list:
            y = all_y_pred.pop(0)
            y = list(y)

            last_y = 0
            last_span_start = 0
            last_span_end = 0
            last_id_start = 0
            last_id_end = 0

            token_id = 0

            for token, span in zip(sentence[0], sentence[1]):
                if len(token) > 0:
                    y_ins = y.pop(0)

                    if last_y != y_ins:
                        if last_y != 0:
                            all_annotation.append(
                                [sent_id, last_span_start, last_span_end, last_id_start, last_id_end, last_y])
                        last_span_start = span[0]
                        last_span_end = span[1]
                        last_id_start = token_id
                        last_id_end = token_id
                        last_y = tag_map[y_ins]
                    else:
                        last_span_end = span[1]
                        last_id_end = token_id

                token_id += 1

            if last_y != 0:
                all_annotation.append([sent_id, last_span_start, last_span_end, last_id_start, last_id_end, last_y])

            sent_id += 1

        with open(output_path + txt_filename[:-3] + 'con', 'w') as writer:
            for ann in all_annotation:
                text = 'c="'
                text += re.sub('\ +', ' ', useful_sentences[ann[0]][ann[1]:ann[2]]).lower()
                text += '" '
                text += str(ann[0] + 1) + ':' + str(ann[3]) + ' ' + str(ann[0] + 1) + ':' + str(ann[4])
                text += '||t="'
                text += all_concept[ann[5]]
                text += '"\n'

                # print(text)
                writer.write(text)

    return output_path


def eval(pred_dir):
    """
    courtesy of https://github.com/text-machine-lab/CliNER
    """
    test_text_path = '../data/raw/test/concept/'

    eval_jar = './i2b2va-eval.jar'

    cmd = 'java -jar %s -rcp %s -scp %s -ft con -ex all' % (eval_jar, test_text_path, pred_dir)
    status = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    result = status.stdout.decode()
    if args.full_report:
        print(result)
    else:
        print(result.split('\n')[7])

    shutil.rmtree(pred_dir)


def main():
    if args.ensemble:
        all_y_pred_list = []
        for file in glob(args.save_path + '*.pkl'):
            all_y_pred, all_l = pickle.load(open(file, 'rb'))
            all_y_pred_list.append(all_y_pred)
        # build ensemble model
        all_y_pred = []
        for i in range(len(all_l)):
            best_v, _ = mode([all_y_pred_list[cv][i] for cv in range(len(all_y_pred_list))], axis=0)
            all_y_pred.append(best_v[0])

    else:
        all_y_pred, all_l = pickle.load(open(args.pickle_file, 'rb'))
    pred_dir = convert_i2b2_format(all_y_pred, all_l)
    eval(pred_dir)


if __name__ == '__main__':
    main()
