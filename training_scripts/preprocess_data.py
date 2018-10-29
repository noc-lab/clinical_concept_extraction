import os
import re
import tarfile
import shutil
import subprocess
import pickle


def unzip_files():
    tar = tarfile.open('../data/raw/concept_assertion_relation_training_data.tar.gz', "r:gz")
    tar.extractall('../data/raw/')
    tar.close()

    tar = tarfile.open('../data/raw/reference_standard_for_test_data.tar.gz', "r:gz")
    tar.extractall('../data/raw/')
    tar.close()

    tar = tarfile.open('../data/raw/test_data.tar.gz', "r:gz")
    tar.extractall('../data/raw/')
    tar.close()

    shutil.move('../data/raw/concept_assertion_relation_training_data', '../data/raw/train')
    shutil.move('../data/raw/reference_standard_for_test_data', '../data/raw/test')
    shutil.move('../data/raw/test_data', '../data/raw/test/txt')
    shutil.move('../data/raw/test/concepts', '../data/raw/test/concept')


def fix_typos():
    """
    credit from https://github.com/wboag/awecm/blob/master/code/eval/concept_extraction/readme.txt
    :return:
    """
    subprocess.call(['sed', '-i',
                     '/c="bun" 30:6 30:6||t="test"/d',
                     '../data/raw/train/partners/concept/920798564.con'])

    subprocess.call(['sed', '-i',
                     '/c="patient" 140:1 140:1||t="test"/d',
                     '../data/raw/train/beth/concept/record-124.con'])


def parse_dir(base_path):
    base_txt_path = base_path + 'txt/'
    base_con_path = base_path + 'concept/'

    all_txt_files = os.listdir(base_txt_path)
    all_txt_files = [item for item in all_txt_files if item[-3:] == 'txt']
    all_txt_files.sort()

    all_tokens = []
    all_concepts = []

    for txt_filename in all_txt_files:
        # read text file
        text = open(base_txt_path + txt_filename, 'r', encoding='utf-8-sig').read()
        token_list = [re.split('\ +', sentence) for sentence in text.split('\n')]
        token_list = [sentence for sentence in token_list if len(sentence) > 0]

        # read concept file
        concepts = open(base_con_path + txt_filename[:-3] + 'con', 'r', encoding='utf-8-sig').read()
        concepts = concepts.split('\n')
        concepts = [concept_item for concept_item in concepts if len(concept_item) > 1]

        # build annotation
        concepts_list = [[''] * len(sentence) for sentence in token_list]

        for concept_item in concepts:
            concept_name = re.findall(r'c="(.*?)" \d', concept_item)[0]
            concept_tag = re.findall(r't="(.*?)"$', concept_item)[0]

            concept_span_string = re.findall(r'(\d+:\d+\ \d+:\d+)', concept_item)[0]

            span_1, span_2 = concept_span_string.split(' ')
            line1, start = span_1.split(':')
            line2, end = span_2.split(':')

            assert line1 == line2

            line1, start, end = int(line1), int(start), int(end)

            concept_name = re.sub(r'\ +', ' ', concept_name)
            original_text = ' '.join(token_list[line1 - 1][start:end + 1])

            if concept_name != original_text.lower():
                print(concept_name, original_text)
                raise RuntimeError

            first = True
            for start_id in range(start, end + 1):
                if first:
                    concepts_list[line1 - 1][start_id] = 'B-' + concept_tag
                    first = False
                else:
                    concepts_list[line1 - 1][start_id] = concept_tag

        all_tokens += (token_list)
        all_concepts += (concepts_list)

    return all_tokens, all_concepts


def main():
    unzip_files()
    fix_typos()

    save_dir = '../data/preprocessed/pkl/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    beth_path = '../data/raw/train/beth/'
    partners_path = '../data/raw/train/partners/'
    text_path = '../data/raw/test/'

    all_tokens, all_concepts = parse_dir(beth_path)
    pickle.dump([all_tokens, all_concepts], open(save_dir + 'beth.pkl', 'wb'))

    all_tokens, all_concepts = parse_dir(partners_path)
    pickle.dump([all_tokens, all_concepts], open(save_dir + 'partners.pkl', 'wb'))

    all_tokens, all_concepts = parse_dir(text_path)
    pickle.dump([all_tokens, all_concepts], open(save_dir + 'text.pkl', 'wb'))

if __name__ == '__main__':
    main()
