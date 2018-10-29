import os
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_float('dropout_rate', 0.5, 'the dropout rate of the CNN or RNN')
flags.DEFINE_string('rnn_cell_type', 'lstm', 'Type of RNN cell used')
flags.DEFINE_integer('hidden_state', 256, 'Number of hidden state')
flags.DEFINE_integer('depth', 2, 'Depth of rnn models')


def bidirectional_rnn_func(x, l, train=True):
    rnn_type = FLAGS.rnn_cell_type
    if rnn_type.lower() == 'lstm':
        rnn_func = tf.nn.rnn_cell.BasicLSTMCell
    elif rnn_type.lower() == 'simplernn':
        rnn_func = tf.nn.rnn_cell.BasicRNNCell
    elif rnn_type.lower() == 'gru':
        rnn_func = tf.nn.rnn_cell.GRUCell
    else:
        raise TypeError

    all_fw_cells = []
    all_bw_cells = []
    for _ in range(FLAGS.depth):
        fw_cell = rnn_func(num_units=FLAGS.hidden_state)
        bw_cell = rnn_func(num_units=FLAGS.hidden_state)
        if train:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, state_keep_prob=FLAGS.dropout_rate)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, state_keep_prob=FLAGS.dropout_rate)
            all_fw_cells.append(fw_cell)
            all_bw_cells.append(bw_cell)
        else:
            all_fw_cells.append(fw_cell)
            all_bw_cells.append(bw_cell)

    rnn_fw_cells = tf.nn.rnn_cell.MultiRNNCell(all_fw_cells)
    rnn_bw_cells = tf.nn.rnn_cell.MultiRNNCell(all_bw_cells)

    rnn_layer, _ = tf.nn.bidirectional_dynamic_rnn(
        rnn_fw_cells, rnn_bw_cells, x, sequence_length=l, dtype=tf.float32)

    rnn_output = tf.concat(rnn_layer, axis=-1)

    return rnn_output


def bidirectional_lstm_func_freeze(x, l):
    all_fw_cells = []
    all_bw_cells = []
    for _ in range(2):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)

        all_fw_cells.append(fw_cell)
        all_bw_cells.append(bw_cell)

    rnn_fw_cells = tf.nn.rnn_cell.MultiRNNCell(all_fw_cells)
    rnn_bw_cells = tf.nn.rnn_cell.MultiRNNCell(all_bw_cells)

    rnn_layer, _ = tf.nn.bidirectional_dynamic_rnn(
        rnn_fw_cells, rnn_bw_cells, x, sequence_length=l, dtype=tf.float32)

    rnn_output = tf.concat(rnn_layer, axis=-1)

    return rnn_output