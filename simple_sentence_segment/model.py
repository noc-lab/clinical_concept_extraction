from six.moves import range
from six.moves import cPickle

import six
import os
import re
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

if six.PY3:
  cls = cPickle.load(open(os.path.join(dir_path, 'model/cls.pkl'), 'rb'), encoding='latin1')
elif six.PY2:
  cls = cPickle.load(open(os.path.join(dir_path, 'model/cls.pkl'), 'rb'))
else:
  raise RuntimeError

char_list = ['x', 'X', 'S', '8', '\n', '.', ':', '-', '*',
             ')', '?', '(', ',', '/', '#', '%', '\t', '+',
             ';', '=', '>', "'", '"', '&', ']', '<']
char2id = {item: item_id for item_id, item in enumerate(char_list)}

DEFAULT_EXCLUSIVE = ['M.D.', 'Dr.', 'vs.']


def get_possible_eos(text, exclusive_phrase):
  possible_eos_re = [' [A-Z]', '\.', '\?', '\n', '\t', '\)',
                     '\]', '\}', '\*', '"', ':']
  eos_re = re.compile('|'.join(possible_eos_re))
  eos = set()
  for eos_find in eos_re.finditer(text):
    start_id = eos_find.span()[0]

    exclusive = False
    for phrase in exclusive_phrase:
      if text[start_id - len(phrase) + 1: start_id + 1] == phrase:
        exclusive = True
        break
    if not exclusive:
      eos.update([start_id])

  eos = list(eos)
  eos.sort()

  return eos


def get_context_char(text, char_id, window=5):
  max_len = len(text)
  assert 0 <= char_id < max_len
  left_text = []
  for i in range(window):
    if char_id - i - 1 < 0:
      left_text.insert(0, ' ')
    else:
      left_text.insert(0, text[char_id - i - 1])

  right_text = []
  for i in range(window):
    if char_id + 1 + i >= max_len:
      right_text.append(' ')
    else:
      right_text.append(text[char_id + 1 + i])

  return left_text + [text[char_id]] + right_text


def one_hot_encoder(X):
  final_output = []
  for i in range(11):
    targets = np.array(X[:, i]).reshape(-1)
    final_output.append(np.eye(27)[targets])
  final_output = np.concatenate(final_output, axis=-1)
  return final_output


def encode_char(c):
  if c.isalpha():
    if c.islower():
      normalized_char = 'x'
    else:
      normalized_char = 'X'
  elif c.isdigit():
    normalized_char = '8'
  elif c == ' ':
    normalized_char = 'S'
  else:
    normalized_char = c

  if normalized_char in char_list:
    return char2id[normalized_char]
  else:
    return 26


def get_span(eos_list, text):
  eos_list = [item + 1 for item in eos_list]
  eos_list.sort()
  eos_list.insert(0, 0)
  if len(text) not in eos_list:
    eos_list.append(len(text))

  spans = []
  for i in range(len(eos_list) - 1):
    s, t = eos_list[i], eos_list[i + 1]
    if len(text[s:t].strip()) > 0:
      spans.append((s, t))
  return spans


def sentence_segment(text, exclusive_phrase=None):
  if exclusive_phrase is None:
    exclusive_phrase = DEFAULT_EXCLUSIVE
  eos_id_list = get_possible_eos(text, exclusive_phrase)

  X = []

  for char_id in eos_id_list:
    features = []
    for c in get_context_char(text, char_id):
      features.append(encode_char(c))

    X.append(features)

  X = np.array(X, dtype=int)

  X = one_hot_encoder(X)
  y = cls.predict(X)

  valid_eos = [x_ for x_, y_ in zip(eos_id_list, y) if y_ == 1]

  all_span = get_span(valid_eos, text)
  return all_span
