import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_processing import MAXLIFE


def dense_layer(x, size,activation_fn, batch_norm = False,phase=False, drop_out=False, keep_prob=None, scope="fc_layer"):
    """
    Helper function to create a fully connected layer with or without batch normalization or dropout regularization

    :param x: previous layer
    :param size: fully connected layer size
    :param activation_fn: activation function
    :param batch_norm: bool to set batch normalization
    :param phase: if batch normalization is set, then phase variable is to mention the 'training' and 'testing' phases
    :param drop_out: bool to set drop-out regularization
    :param keep_prob: if drop-out is set, then to mention the keep probability of dropout
    :param scope: variable scope name
    :return: fully connected layer
    """
    with tf.variable_scope(scope):
        if batch_norm:
            dence_layer = tf.contrib.layers.fully_connected(x, size, activation_fn=None)
            dence_layer_bn = BatchNorm(name="batch_norm_" + scope)(dence_layer, train=phase)
            return_layer = activation_fn(dence_layer_bn)
        else:
            return_layer = tf.layers.dense(x, size,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation=activation_fn)
        if drop_out:
            return_layer = tf.nn.dropout(return_layer, keep_prob)

        return return_layer


def get_RNNCell(cell_types, keep_prob, state_size, build_with_dropout=True):
    """
    Helper function to get a different types of RNN cells with or without dropout wrapper
    :param cell_types: cell_type can be 'GRU' or 'LSTM' or 'LSTM_LN' or 'GLSTMCell' or 'LSTM_BF' or 'None'
    :param keep_prob: dropout keeping probability
    :param state_size: number of cells in a layer
    :param build_with_dropout: to enable the dropout for rnn layers
    :return:
    """
    cells = []
    for cell_type in cell_types:
        if cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=state_size,
                                          bias_initializer=tf.zeros_initializer())  # Or GRU(num_units)
        elif cell_type == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(num_units=state_size, use_peepholes=True, state_is_tuple=True,
                                           initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_LN':
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(state_size)
        elif cell_type == 'GLSTMCell':
            cell = tf.contrib.rnn.GLSTMCell(num_units=state_size, initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_BF':
            cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=state_size, use_peephole=True)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

        if build_with_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    if build_with_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


class BatchNorm(object):
    """
    usage : dence_layer_bn = BatchNorm(name="batch_norm_" + scope)(previous_layer, train=is_train)
    """
    def __init__(self, epsilon=1e-5, momentum=0.999, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def batch_generator(x_train, y_train, batch_size, sequence_length, online=False, online_shift=1):
    """
    Generator function for creating random batches of training-data for many to many models
    """
    num_x_sensors = x_train.shape[1]
    num_train = x_train.shape[0]
    idx = 0

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_sensors)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)
        # print(idx)
        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            if online == True and (idx >= num_train or (idx + sequence_length) > num_train):
                idx = 0
            elif online == False:
                idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train[idx:idx + sequence_length]
            y_batch[i] = y_train[idx:idx + sequence_length]
            # print(i,idx)
            if online:
                idx = idx + online_shift  # check if its nee to be idx=idx+1
                # print(idx)
        # print(idx)
        yield (x_batch, y_batch)


def trjectory_generator(x_train, y_train, test_engine_id, sequence_length, graph_batch_size, lower_bound):
    """
    Extract training trjectories one by one
    test_engine_id = [11111111...,22222222....,...]
    """
    DEBUG = False
    num_x_sensors = x_train.shape[1]
    idx = 0
    engine_ids = test_engine_id.unique()
    if DEBUG: print("total trjectories: ", len(engine_ids))

    while True:
        for id in engine_ids:

            indexes = test_engine_id[test_engine_id == id].index
            training_data = x_train[indexes]
            if DEBUG: print("engine_id: ", id, "start", indexes[0], "end", indexes[-1], "trjectory_len:", len(indexes))
            batch_size = int(training_data.shape[0] / sequence_length) + 1
            idx = indexes[0]

            x_batch = np.zeros(shape=(batch_size, sequence_length, num_x_sensors), dtype=np.float32)
            y_batch = np.zeros(shape=(batch_size, sequence_length), dtype=np.float32)

            for i in range(batch_size):

                # Copy the sequences of data starting at this index.
                if DEBUG: print("current idx=", idx)
                if idx >= x_train.shape[0]:
                    if DEBUG: print("BREAK")
                    break
                elif (idx + sequence_length) > x_train.shape[0]:
                    if DEBUG: print("BREAK", idx, x_train.shape[0], idx + sequence_length - x_train.shape[0])
                    x_tmp = x_train[idx:]
                    y_tmp = y_train[idx:]
                    remain = idx + sequence_length - x_train.shape[0]
                    x_batch[i] = np.concatenate((x_tmp, x_train[0:remain]))
                    y_batch[i] = np.concatenate((y_tmp, y_train[0:remain]))
                    break

                x_batch[i] = x_train[idx:idx + sequence_length]

                if idx > indexes[-1] - sequence_length:
                    y_tmp = np.copy(y_train[idx:idx + sequence_length])
                    remain = sequence_length - (indexes[-1] - idx + 1)  # abs(training_data.shape[0]-sequence_length)
                    if DEBUG: print("(idx + sequence_length) > trj_len:", "remain", remain)
                    y_tmp[-remain:] = lower_bound
                    y_batch[i] = y_tmp
                else:
                    y_batch[i] = y_train[idx:idx + sequence_length]

                idx = idx + sequence_length

            batch_size_gap = graph_batch_size - x_batch.shape[0]
            if batch_size_gap > 0:
                for i in range(batch_size_gap):
                    x_tmp = -0.01 * np.ones(shape=(sequence_length, num_x_sensors), dtype=np.float32)
                    y_tmp = -0.01 * np.ones(shape=(sequence_length), dtype=np.float32)
                    xx = np.append(x_batch, x_tmp)
                    x_batch = np.reshape(xx, [x_batch.shape[0] + 1, x_batch.shape[1], x_batch.shape[2]])
                    yy = np.append(y_batch, y_tmp)
                    y_batch = np.reshape(yy, [y_batch.shape[0] + 1, x_batch.shape[1]])
            yield (x_batch, y_batch)


def plot_data(data, label=""):
    """
    Plot every plot on top of each other
    """
    from matplotlib import pyplot as plt
    if type(data) is list:
        for x in data:
            plt.plot(x, label=label)
    else:
        plt.plot(data, label=label)
    plt.show()


def model_summary(learning_rate,batch_size,lstm_layers,lstm_layer_size,fc_layer_size,sequence_length,n_channels,path_checkpoint,spacial_note=''):
    path_checkpoint=path_checkpoint + ".txt"
    if not os.path.exists(os.path.dirname(path_checkpoint)):
        os.makedirs(os.path.dirname(path_checkpoint))

    with open(path_checkpoint, "w") as text_file:
        variables = tf.trainable_variables()

        print('---------', file=text_file)
        print(path_checkpoint, file=text_file)
        print(spacial_note, file=text_file)
        print('---------', '\n', file=text_file)

        print('---------', file=text_file)
        print('MAXLIFE: ', MAXLIFE,'\n',  file=text_file)
        print('learning_rate: ', learning_rate, file=text_file)
        print('batch_size: ', batch_size, file=text_file)
        print('lstm_layers: ', lstm_layers, file=text_file)
        print('lstm_layer_size: ', lstm_layer_size, file=text_file)
        print('fc_layer_size: ', fc_layer_size, '\n', file=text_file)
        print('sequence_length: ', sequence_length, file=text_file)
        print('n_channels: ', n_channels, file=text_file)
        print('---------', '\n', file=text_file)

        print('---------', file=text_file)
        print('Variables: name (type shape) [size]', file=text_file)
        print('---------', '\n', file=text_file)
        total_size = 0
        total_bytes = 0
        for var in variables:
            # if var.num_elements() is None or [] assume size 0.
            var_size = var.get_shape().num_elements() or 0
            var_bytes = var_size * var.dtype.size
            total_size += var_size
            total_bytes += var_bytes
            print(var.name, slim.model_analyzer.tensor_description(var), '[%d, bytes: %d]' %
                      (var_size, var_bytes), file=text_file)

        print('\nTotal size of variables: %d' % total_size, file=text_file)
        print('Total bytes of variables: %d' % total_bytes, file=text_file)


def scoring_func(error_arr):
    '''

    :param error_arr: a list of errors for each training trajectory
    :return: standered score value for RUL
    '''
    import math
    # print(error_arr)
    pos_error_arr = error_arr[error_arr >= 0]
    neg_error_arr = error_arr[error_arr < 0]

    score = 0
    # print(neg_error_arr)
    for error in neg_error_arr:
        score = math.exp(-(error / 13)) - 1 + score
        # print(math.exp(-(error / 13)),score,error)

    # print(pos_error_arr)
    for error in pos_error_arr:
        score = math.exp(error / 10) - 1 + score
        # print(math.exp(error / 10),score, error)
    return score


def conv_layer(X,filters,kernel_size,strides,padding,batch_norm,is_train,scope):
    """
    1D convolutional layer with or without dropout or batch normalization

    :param batch_norm:  bool, enable batch normalization
    :param is_train: bool, mention if current phase is training phase
    :param scope: variable scope
    :return: 1D-convolutional layer
    """
    with tf.variable_scope(scope):
        if batch_norm:
            conv1 = tf.layers.conv1d(inputs=X, filters=filters, kernel_size=kernel_size, strides=strides,
                                     padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer())
            return tf.nn.relu(BatchNorm(name="norm_"+scope)(conv1, train=is_train))
        else:
            return tf.layers.conv1d(inputs=X, filters=filters, kernel_size=kernel_size, strides=strides,
                                     padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     activation=tf.nn.relu)


def get_predicted_expected_RUL(__y, __y_pred, lower_bound=-1):
    trj_end = np.argmax(__y == lower_bound) - 1
    trj_pred = __y_pred[:trj_end]
    trj_pred[trj_pred < 0] = 0
    # if trj_pred[-1] < 0: print(trj_pred[-1])
    RUL_predict = round(trj_pred[-1], 0)
    RUL_expected = round(__y[trj_end], 0)

    return RUL_predict, RUL_expected
