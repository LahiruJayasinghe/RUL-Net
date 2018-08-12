import os
import pickle
import numpy as np
import tensorflow as tf

DEBUG = False


def cache(cache_path, obj=None):
    if os.path.exists(cache_path):
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)
        print("- Data loaded from cache-file: " + cache_path)
    elif obj is not None:
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)
        print("- Data saved to cache-file: " + cache_path)
    return obj


def movingavg(data, window):  # [n_samples, n_features]
    data_new = np.transpose(data)
    if data_new.ndim > 1:
        tmp = []
        for i in range(data_new.shape[0]):
            ma = movingavg(np.squeeze(data_new[i]), window)
            tmp.append(ma)
        smas = np.array(tmp)
    else:
        w = np.repeat(1.0, window) / window
        smas = np.convolve(data_new, w, 'valid')
    smas = np.transpose(smas)
    return smas  # [n_samples, n_features]


def get_weights(shape, scope_name):
    with tf.variable_scope(scope_name):
        w = tf.get_variable(scope_name + '_w', shape, initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(scope_name + '_b', [shape[-1]], initializer=tf.zeros_initializer(), dtype=tf.float32)
        return w, b


def apply_conv(x, kernel_height, kernel_width, num_channels, depth, scope_name):
    weights, biases = get_weights([kernel_height, kernel_width, num_channels, depth], scope_name)
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding="VALID"), biases))


def apply_max_pool(x, kernel_height, kernel_width, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, 1, stride_size, 1], padding="VALID")


def apply_conv_new(x, kernel_height, kernel_width, num_channels, depth, scope_name):
    W, b = get_weights([kernel_height, kernel_width, num_channels, depth], scope_name)
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def apply_max_pool_new(x, kernel_height, kernel_width, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, stride_size, stride_size, 1],
                          padding='SAME')


def apply_avg_pool_new(x, kernel_height, kernel_width, stride_size):
    return tf.nn.avg_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, stride_size, stride_size, 1],
                          padding='SAME')


def get_RNNCell(cell_types, keep_prob, state_size, build_with_dropout=True):
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


def get_inception1D(X):

    conv1_11 = tf.layers.conv1d(inputs=X, filters=36, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)
    conv1_21 = tf.layers.conv1d(inputs=X, filters=18, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)
    conv1_31 = tf.layers.conv1d(inputs=X, filters=18, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)
    avg_pool_41 = tf.layers.average_pooling1d(inputs=X, pool_size=2, strides=1, padding='same')
    conv2_22 = tf.layers.conv1d(inputs=conv1_21, filters=36, kernel_size=2, strides=1,
                                padding='same', activation=tf.nn.relu)
    conv4_32 = tf.layers.conv1d(inputs=conv1_31, filters=36, kernel_size=4, strides=1,
                                padding='same', activation=tf.nn.relu)
    conv1_42 = tf.layers.conv1d(inputs=avg_pool_41, filters=36, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu)

    inception_out = tf.concat([conv1_11, conv2_22, conv4_32, conv1_42], axis=2)
    return inception_out


def windows(nrows, size):
    start, step = 0, 2
    while start < nrows:
        # print(start,nrows)
        yield start, start + size
        start += step


def segment_signal(features, labels, window_size=15):
    segments = np.empty((0, window_size))
    segment_labels = np.empty((0))
    nrows = len(features)
    for (start, end) in windows(nrows, window_size):
        if (len(features.iloc[start:end]) == window_size):
            segment = features[start:end].T  # Transpose to get segment of size 24 x 15
            label = labels[(end - 1)]
            segments = np.vstack([segments, segment])
            segment_labels = np.append(segment_labels, label)
    segments = segments.reshape(-1, 24, window_size, 1)  # number of features  = 24
    segment_labels = segment_labels.reshape(-1, 1)
    return segments, segment_labels


def batch_generator_CNN(x_train, y_train, batch_size, sequence_length, online=False, shift=1):
    """
    Generator function for creating random batches of training-data for many to one models
    """
    num_x_sensors = x_train.shape[1]
    num_train = x_train.shape[0]
    idx = 0

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, num_x_sensors, sequence_length)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)
        # print(idx)
        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        if online == False:
            idx = np.random.randint(num_train - sequence_length - batch_size)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            if online == True and (idx >= num_train or (idx + sequence_length) > num_train):
                # print("batach_size:",batch_size,"last idx:",idx,"now idx AT THE BEGINING")
                idx = 0

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train[idx:idx + sequence_length].T
            # y_batch[i] = y_train[idx:idx + sequence_length]
            y_batch[i] = y_train[idx + sequence_length - 1]

            idx = idx + shift  # check if its nee to be idx=idx+1
            # print(idx)

        yield (x_batch[:, ..., np.newaxis], y_batch[:, ..., np.newaxis])


def batch_generator(x_train, y_train, batch_size, sequence_length, online=False, online_shift=1):
    """
    Generator function for creating random batches of training-data for meny to meny models
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
    """
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
