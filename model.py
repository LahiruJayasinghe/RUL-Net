from matplotlib import pyplot as plt
import time
import datetime
from utils_laj import *
from data_processing import get_CMAPSSData, get_PHM08Data, data_augmentation, analyse_Data

today = datetime.date.today()


def CNNLSTM(dataset, Train=False, trj_wise=False, plot=False):
    '''
    The architecture is a Meny-to-meny model combining CNN and LSTM models
    :param dataset: select the specific dataset between PHM08 or CMAPSS
    :param Train: select between training and testing
    :param trj_wise: Trajectorywise calculate RMSE and scores
    '''

    #### checkpoint saving path ####
    # path_checkpoint = '.\Save\Save_CNN\EX11_CNNLSTM_ML035_KR130250_kinkRUL_FD004\CNN1DLSTM_3n2_layers'
    # path_checkpoint = '.\Save\Save_CNN\EX12_CNNLSTM_ML035_KR130250_kinkRUL_FD001_aug\CNN1DLSTM_3n2_layers'
    path_checkpoint = '.\Save\Save_CNN\EX13_CNNLSTM_ML120_GRAD1_kinkRUL_FD003\CNN1DLSTM_3n2_layers'
    # path_checkpoint = '.\Save\Save_CNN\experiment_testing_models_CMAPSS_kinkRUL\CNN1DLSTM_3n2_layers'
    ################################

    if dataset == "cmapss":
        training_data, testing_data, training_pd, testing_pd = get_CMAPSSData(save=False)
        x_train = training_data[:, :training_data.shape[1] - 1]
        y_train = training_data[:, training_data.shape[1] - 1]
        print("training data CNNLSTM: ", x_train.shape, y_train.shape)

        x_test = testing_data[:, :testing_data.shape[1] - 1]
        y_test = testing_data[:, testing_data.shape[1] - 1]
        print("testing data CNNLSTM: ", x_test.shape, y_test.shape)

    elif dataset == "phm":
        training_data, testing_data, phm_testing_data = get_PHM08Data(save=False)
        x_validation = phm_testing_data[:, :phm_testing_data.shape[1] - 1]
        y_validation = phm_testing_data[:, phm_testing_data.shape[1] - 1]
        print("testing data: ", x_validation.shape, y_validation.shape)

    batch_size = 1024  # Batch size
    if Train == False: batch_size = 5

    sequence_length = 100  # Number of steps
    learning_rate = 0.001  # 0.0001
    epochs = 5000
    ann_hidden = 50

    n_channels = 24

    lstm_size = n_channels * 3  # 3 times the amount of channels
    num_layers = 2  # Number of layers

    X = tf.placeholder(tf.float32, [None, sequence_length, n_channels], name='inputs')
    Y = tf.placeholder(tf.float32, [None, sequence_length], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    # (batch, 128, 9) --> (batch, 128, 18)

    conv1 = tf.layers.conv1d(inputs=X, filters=18, kernel_size=2, strides=1,
                             padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    # (batch, 64, 18) --> (batch, 32, 36)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1,
                             padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    # (batch, 32, 36) --> (batch, 16, 72)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1,
                             padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    conv_last_layer = max_pool_3

    shape = conv_last_layer.get_shape().as_list()
    CNN_flat = tf.reshape(conv_last_layer, [-1, shape[1] * shape[2]])

    dence_layer_1 = tf.layers.dense(CNN_flat, sequence_length * n_channels,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
    dence_layer_1_dropout = tf.nn.dropout(dence_layer_1, keep_prob)

    lstm_input = tf.reshape(dence_layer_1_dropout, [-1, sequence_length, n_channels])

    cell = get_RNNCell(['LSTM'] * num_layers, keep_prob=keep_prob, state_size=lstm_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_output, states = tf.nn.dynamic_rnn(cell, lstm_input, dtype=tf.float32, initial_state=init_state)
    stacked_rnn_output = tf.reshape(rnn_output, [-1, lstm_size])  # change the form into a tensor
    y_flat = tf.reshape(Y, [-1])

    dence_layer_2 = tf.layers.dense(stacked_rnn_output, ann_hidden,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
    dence_layer_dropout_2 = tf.nn.dropout(dence_layer_2, keep_prob)

    prediction = tf.layers.dense(dence_layer_dropout_2, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
    prediction = tf.reshape(prediction, [-1])

    h = prediction - y_flat
    # cost_function = tf.reduce_mean(tf.square(prediction - y_flat))
    cost_function = tf.reduce_sum(tf.square(prediction - y_flat))
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(h * RESCALE)))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)

    saver = tf.train.Saver()
    training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
    testing_generator = batch_generator(x_test, y_test, batch_size, sequence_length, online=False)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        if Train == True:
            cost = []
            iteration = int(x_train.shape[0] / batch_size)
            print("Training set MSE")
            print("No epoches: ", epochs, "No itr: ", iteration)
            __start = time.time()
            for ep in range(epochs):

                for itr in range(iteration):
                    ## training ##
                    batch_x, batch_y = next(training_generator)
                    session.run(optimizer,
                                feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8, learning_rate_: learning_rate})
                    cost.append(
                        RMSE.eval(feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0, learning_rate_: learning_rate}))
                    # fid =  {X: batch_x, Y: batch_y, keep_prob: 0.8, learning_rate_: learning_rate}
                    # rnn_output_t = rnn_output.eval(feed_dict=fid)
                    # lstm_input_t = lstm_input.eval(feed_dict=fid)
                    # lstm_in_t = lstm_input.eval(feed_dict=fid)
                    # rnn_output_t = rnn_output.eval(feed_dict=fid)
                    # stacked_rnn_output_t = stacked_rnn_output.eval(feed_dict=fid)
                    # y_flat_t = y_flat.eval(feed_dict=fid)
                    # # stacked_rnn_output_t = y_.eval(feed_dict=fid)

                    # print("rnn_output", rnn_output_t.shape)
                    # print("lstm_input", lstm_input_t.shape)
                    # print("lstm_in_t", lstm_in_t.shape)
                    # print("rnn_output_t", rnn_output_t.shape)
                    # print("stacked_rnn_output_t", stacked_rnn_output_t.shape)
                    # print("y_flat_t",y_flat_t.shape)
                    # print("stacked_rnn_output_t", stacked_rnn_output_t.shape)

                    # exit(0)
                    ##############
                x_test_batch, y_test_batch = next(testing_generator)
                mse_train, rmse_train = session.run([cost_function, RMSE],
                                                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0,
                                                               learning_rate_: learning_rate})
                mse_test, rmse_test = session.run([cost_function, RMSE],
                                                  feed_dict={X: x_test_batch, Y: y_test_batch, keep_prob: 1.0,
                                                             learning_rate_: learning_rate})

                time_per_ep = (time.time() - __start)
                time_remaining = ((epochs - ep) * time_per_ep) / 3600
                print("CNNLSTM", "epoch:", ep, "\tTrainig-",
                      "MSE:", mse_train, "RMSE:", rmse_train, "\tTesting-", "MSE", mse_test, "RMSE", rmse_test,
                      "\ttime/epoch:", round(time_per_ep, 2), "\ttime_remaining: ",
                      int(time_remaining), " hr:", round((time_remaining % 1) * 60, 1), " min", "\ttime_stamp: ",
                      datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
                __start = time.time()

                if ep % 10 == 0 and ep != 0:
                    save_path = saver.save(session, path_checkpoint)
                    if os.path.exists(path_checkpoint + '.meta'):
                        print("Model saved to file: %s" % path_checkpoint)
                    else:
                        print("NOT SAVED!!!", path_checkpoint)

                if ep % 1000 == 0 and ep != 0: learning_rate = learning_rate / 10

                # ######## Online augmentation #########
                # data_augmentation(files=1)
                # training_data, testing_data, training_pd, testing_pd = get_CMAPSSData(save=False)
                # x_train = training_data[:, :training_data.shape[1] - 1]
                # y_train = training_data[:, training_data.shape[1] - 1]
                # training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
                # ######################################

            save_path = saver.save(session, path_checkpoint)
            if os.path.exists(path_checkpoint + '.meta'):
                print("Model saved to file: %s" % path_checkpoint)
            else:
                print("NOT SAVED!!!", path_checkpoint)
            plt.plot(cost)
            plt.show()
        else:
            saver.restore(session, path_checkpoint)
            print("Model restored from file: %s" % path_checkpoint)

            if trj_wise:
                trj_iteration = len(test_engine_id.unique())
                print("total trajectories: ", trj_iteration)
                error_list = []
                pred_list = []
                expected_list = []
                lower_bound = -0.01
                test_trjectory_generator = trjectory_generator(x_test, y_test, test_engine_id, sequence_length,
                                                               batch_size, lower_bound)
                for itr in range(trj_iteration):
                    trj_x, trj_y = next(test_trjectory_generator)

                    __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                       feed_dict={X: trj_x, Y: trj_y, keep_prob: 1.0})
                    trj_end = np.argmax(__y == lower_bound) - 1
                    trj_pred = __y_pred[:trj_end]
                    # RUL_predict = __y_pred[np.argmin(trj_pred)] * RESCALE
                    RUL_predict = round(trj_pred[-1] * RESCALE, 0)
                    RUL_expected = round(__y[trj_end] * RESCALE, 0)
                    error_list.append(RUL_predict - RUL_expected)
                    pred_list.append(RUL_predict)
                    expected_list.append(RUL_expected)
                    if RUL_expected <= 0: continue
                    print("id: ", itr + 1, "expected: ", RUL_expected, "\t", "predict: ", RUL_predict, "\t", "error: ",
                          RUL_predict - RUL_expected, "true_error: ", RUL_predict - true_rul[itr])
                    # plt.plot(__y_pred* RESCALE, label="prediction")
                    # plt.plot(__y* RESCALE, label="expected")
                    # plt.show()
                error_list = np.array(error_list)
                error_list = error_list.ravel()
                rmse = np.sqrt(np.sum(np.square(error_list)) / len(error_list))  # RMSE
                print(rmse, scoring_func(error_list))
                if plot:
                    plt.figure()
                    plt.plot(expected_list, 'o', color='black', label="expected")
                    plt.plot(pred_list, 'o', color='red', label="predicted")
                    plt.legend()
                    plt.show()
                fig, ax = plt.subplots()
                ax.stem(expected_list, linefmt='b-', label="expected")
                ax.stem(pred_list, linefmt='r-', label="predicted")
                plt.legend()
                plt.show()

            else:
                x_validation = x_test
                y_validation = y_test

                validation_generator = batch_generator(x_validation, y_validation, batch_size, sequence_length,
                                                       online=True,
                                                       online_shift=sequence_length)

                full_prediction = []
                actual_rul = []
                error_list = []

                iteration = int(x_validation.shape[0] / (batch_size * sequence_length))
                print("#of validation points:", x_validation.shape[0], "#datapoints covers from minibatch:",
                      batch_size * sequence_length, "iterations/epoch", iteration)

                for itr in range(iteration):
                    x_validate_batch, y_validate_batch = next(validation_generator)
                    __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                       feed_dict={X: x_validate_batch, Y: y_validate_batch,
                                                                  keep_prob: 1.0})
                    full_prediction.append(__y_pred * RESCALE)
                    actual_rul.append(__y * RESCALE)
                    error_list.append(error * RESCALE)
                    # x = np.reshape(x_validate_batch,[x_validate_batch.shape[0]*x_validate_batch.shape[1],x_validate_batch.shape[2]])
                    # y = np.reshape(x_train_batch,
                    #                [x_train_batch.shape[0] * x_train_batch.shape[1], x_train_batch.shape[2]])
                    # plt.figure(1)
                    # plt.plot(__y_pred, label="prediction")
                    # plt.plot(__y, label="expected")
                    # plt.plot(y,label="y")
                    # plt.figure(2)
                    # plt.plot(x,label="x")
                    # plt.legend()
                    # plt.show()
                full_prediction = np.array(full_prediction)
                full_prediction = full_prediction.ravel()
                actual_rul = np.array(actual_rul)
                actual_rul = actual_rul.ravel()
                error_list = np.array(error_list)
                error_list = error_list.ravel()
                rmse = np.sqrt(np.sum(np.square(error_list)) / len(error_list))  # RMSE

                print(y_validation.shape, full_prediction.shape, "RMSE:", rmse, "Score:", scoring_func(error_list))
                if plot:
                    plt.plot(full_prediction, label="prediction")
                    plt.plot(actual_rul, label="expected")
                    plt.legend()
                    plt.show()


def InceptionLSTM(dataset, Train=False, trj_wise=False, plot=False):
    '''
    The architecture is a Meny-to-meny model combining CNN inception module and a LSTM two layered model
    :param dataset: select the specific dataset between PHM08 or CMAPSS
    :param Train: select between training and testing
    :param trj_wise: Trajectorywise calculate RMSE and scores
    '''

    #### checkpoint saving path ####
    path_checkpoint = '.\Save\Save_CNN\experiment_testing_models_CMAPSS_kinkRUL\inception_2_lstm2_layers'
    ################################

    if dataset == "cmapss":
        training_data, testing_data, training_pd, testing_pd = get_CMAPSSData(save=False)
        x_train = training_data[:, :training_data.shape[1] - 1]
        y_train = training_data[:, training_data.shape[1] - 1]
        print("training data InceptionLSTM: ", x_train.shape, y_train.shape)

        x_test = testing_data[:, :testing_data.shape[1] - 1]
        y_test = testing_data[:, testing_data.shape[1] - 1]
        print("testing data InceptionLSTM: ", x_test.shape, y_test.shape)

    elif dataset == "phm":
        training_data, testing_data, phm_testing_data = get_PHM08Data(save=False)
        x_validation = phm_testing_data[:, :phm_testing_data.shape[1] - 1]
        y_validation = phm_testing_data[:, phm_testing_data.shape[1] - 1]
        print("testing data: ", x_validation.shape, y_validation.shape)

    batch_size = 1024  # Batch size
    if Train == False: batch_size = 5

    sequence_length = 100  # Number of steps
    learning_rate = 0.001  # 0.0001
    epochs = 5000
    ann_hidden = 50

    n_channels = 24

    lstm_size = n_channels * 3  # 3 times the amount of channels
    num_layers = 2  # Number of layers

    X = tf.placeholder(tf.float32, [None, sequence_length, n_channels], name='inputs')
    Y = tf.placeholder(tf.float32, [None, sequence_length], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    #### Stream ####
    # (batch, sequence_length, n_channels) --> (batch, 64, 18)
    conv1 = tf.layers.conv1d(inputs=X, filters=18, kernel_size=2, strides=1,
                             padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    # (batch, 64, 18) --> (batch, 32, 18)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=18, kernel_size=2, strides=1,
                             padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    # (batch, 32, 18) --> (batch, 16, 36)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=36, kernel_size=2, strides=1,
                             padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    # (batch, 16, 36) --> (batch, 8, 36)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=36, kernel_size=2, strides=1,
                             padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
    ################

    inception_1 = get_inception1D(max_pool_4)  # inception layer
    inception_2 = get_inception1D(inception_1)  # inception layer
    max_pool_5 = tf.layers.max_pooling1d(inputs=inception_2, pool_size=3, strides=3, padding='same')
    inception_3 = get_inception1D(max_pool_5)  # inception layer

    conv_last_layer = inception_3

    shape = conv_last_layer.get_shape().as_list()
    CNN_flat = tf.reshape(conv_last_layer, [-1, shape[1] * shape[2]])

    dence_layer_1 = tf.layers.dense(CNN_flat, sequence_length * n_channels,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
    dence_layer_1_dropout = tf.nn.dropout(dence_layer_1, keep_prob)

    lstm_input = tf.reshape(dence_layer_1_dropout, [-1, sequence_length, n_channels])

    cell = get_RNNCell(['LSTM'] * num_layers, keep_prob=keep_prob, state_size=lstm_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_output, states = tf.nn.dynamic_rnn(cell, lstm_input, dtype=tf.float32, initial_state=init_state)
    stacked_rnn_output = tf.reshape(rnn_output, [-1, lstm_size])  # change the form into a tensor
    y_flat = tf.reshape(Y, [-1])

    dence_layer_2 = tf.layers.dense(stacked_rnn_output, ann_hidden,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
    dence_layer_dropout_2 = tf.nn.dropout(dence_layer_2, keep_prob)

    prediction = tf.layers.dense(dence_layer_dropout_2, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=None)
    prediction = tf.reshape(prediction, [-1])

    h = prediction - y_flat
    # cost_function = tf.reduce_mean(tf.square(prediction - y_flat))
    cost_function = tf.reduce_sum(tf.square(prediction - y_flat))
    RMSE = tf.sqrt(tf.reduce_mean(tf.square(h * RESCALE)))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost_function)

    saver = tf.train.Saver()
    training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
    testing_generator = batch_generator(x_test, y_test, batch_size, sequence_length, online=False)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        if Train == True:
            cost = []
            iteration = int(x_train.shape[0] / batch_size)
            print("Training set MSE")
            print("No epoches: ", epochs, "No itr: ", iteration)
            __start = time.time()
            for ep in range(epochs):

                for itr in range(iteration):
                    ## training ##
                    batch_x, batch_y = next(training_generator)
                    session.run(optimizer,
                                feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8, learning_rate_: learning_rate})
                    cost.append(
                        RMSE.eval(feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0, learning_rate_: learning_rate}))
                    # fid =  {X: batch_x, Y: batch_y, keep_prob: 0.8, learning_rate_: learning_rate}
                    # rnn_output_t = rnn_output.eval(feed_dict=fid)
                    # lstm_input_t = lstm_input.eval(feed_dict=fid)
                    # lstm_in_t = lstm_input.eval(feed_dict=fid)
                    # rnn_output_t = rnn_output.eval(feed_dict=fid)
                    # stacked_rnn_output_t = stacked_rnn_output.eval(feed_dict=fid)
                    # y_flat_t = y_flat.eval(feed_dict=fid)
                    # # stacked_rnn_output_t = y_.eval(feed_dict=fid)

                    # print("rnn_output", rnn_output_t.shape)
                    # print("lstm_input", lstm_input_t.shape)
                    # print("lstm_in_t", lstm_in_t.shape)
                    # print("rnn_output_t", rnn_output_t.shape)
                    # print("stacked_rnn_output_t", stacked_rnn_output_t.shape)
                    # print("y_flat_t",y_flat_t.shape)
                    # print("stacked_rnn_output_t", stacked_rnn_output_t.shape)

                    # exit(0)
                    ##############
                x_test_batch, y_test_batch = next(testing_generator)
                mse_train, rmse_train = session.run([cost_function, RMSE],
                                                    feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0,
                                                               learning_rate_: learning_rate})
                mse_test, rmse_test = session.run([cost_function, RMSE],
                                                  feed_dict={X: x_test_batch, Y: y_test_batch, keep_prob: 1.0,
                                                             learning_rate_: learning_rate})

                time_per_ep = (time.time() - __start)
                time_remaining = ((epochs - ep) * time_per_ep) / 3600
                print("InceptionLSTM", "epoch:", ep, "\tTrainig-",
                      "MSE:", mse_train, "RMSE:", rmse_train, "\tTesting-", "MSE", mse_test, "RMSE", rmse_test,
                      "\ttime/epoch:", round(time_per_ep, 2), "\ttime_remaining: ",
                      int(time_remaining), " hr:", round((time_remaining % 1) * 60, 1), " min", "\ttime_stamp: ",
                      datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
                __start = time.time()

                if ep % 10 == 0 and ep != 0:
                    save_path = saver.save(session, path_checkpoint)
                    if os.path.exists(path_checkpoint + '.meta'):
                        print("Model saved to file: %s" % path_checkpoint)
                    else:
                        print("NOT SAVED!!!", path_checkpoint)

                if ep % 1000 == 0 and ep != 0: learning_rate = learning_rate / 10

                # ######## Online augmentation #########
                # data_augmentation(files=1)
                # training_data, testing_data, training_pd, testing_pd = get_CMAPSSData(save=False)
                # x_train = training_data[:, :training_data.shape[1] - 1]
                # y_train = training_data[:, training_data.shape[1] - 1]
                # training_generator = batch_generator(x_train, y_train, batch_size, sequence_length, online=True)
                # ######################################

            save_path = saver.save(session, path_checkpoint)
            if os.path.exists(path_checkpoint + '.meta'):
                print("Model saved to file: %s" % path_checkpoint)
            else:
                print("NOT SAVED!!!", path_checkpoint)
            plt.plot(cost)
            plt.show()
        else:
            saver.restore(session, path_checkpoint)
            print("Model restored from file: %s" % path_checkpoint)

            if trj_wise:
                trj_iteration = len(test_engine_id.unique())
                print("total trajectories: ", trj_iteration)
                error_list = []
                pred_list = []
                expected_list = []
                lower_bound = -0.01
                test_trjectory_generator = trjectory_generator(x_test, y_test, test_engine_id, sequence_length,
                                                               batch_size, lower_bound)
                for itr in range(trj_iteration):
                    trj_x, trj_y = next(test_trjectory_generator)

                    __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                       feed_dict={X: trj_x, Y: trj_y, keep_prob: 1.0})
                    trj_end = np.argmax(__y == lower_bound) - 1
                    trj_pred = __y_pred[:trj_end]
                    # RUL_predict = __y_pred[np.argmin(trj_pred)] * RESCALE
                    RUL_predict = round(trj_pred[-1] * RESCALE, 0)
                    RUL_expected = round(__y[trj_end] * RESCALE, 0)
                    error_list.append(RUL_predict - RUL_expected)
                    pred_list.append(RUL_predict)
                    expected_list.append(RUL_expected)
                    if RUL_expected <= 0: continue
                    print("id: ", itr + 1, "expected: ", RUL_expected, "\t", "predict: ", RUL_predict, "\t", "error: ",
                          RUL_predict - RUL_expected, "true_error: ", RUL_predict - true_rul[itr])
                    # plt.plot(__y_pred* RESCALE, label="prediction")
                    # plt.plot(__y* RESCALE, label="expected")
                    # plt.show()
                error_list = np.array(error_list)
                error_list = error_list.ravel()
                rmse = np.sqrt(np.sum(np.square(error_list)) / len(error_list))  # RMSE
                print(rmse, scoring_func(error_list))
                if plot:
                    plt.figure()
                    plt.plot(expected_list, 'o', color='black', label="expected")
                    plt.plot(pred_list, 'o', color='red', label="predicted")
                    plt.legend()
                    plt.show()
                fig, ax = plt.subplots()
                ax.stem(expected_list, linefmt='b-', label="expected")
                ax.stem(pred_list, linefmt='r-', label="predicted")
                plt.legend()
                plt.show()

            else:
                x_validation = x_test
                y_validation = y_test

                validation_generator = batch_generator(x_validation, y_validation, batch_size, sequence_length,
                                                       online=True,
                                                       online_shift=sequence_length)

                full_prediction = []
                actual_rul = []
                error_list = []

                iteration = int(x_validation.shape[0] / (batch_size * sequence_length))
                print("#of validation points:", x_validation.shape[0], "#datapoints covers from minibatch:",
                      batch_size * sequence_length, "iterations/epoch", iteration)

                for itr in range(iteration):
                    x_validate_batch, y_validate_batch = next(validation_generator)
                    __y_pred, error, __y = session.run([prediction, h, y_flat],
                                                       feed_dict={X: x_validate_batch, Y: y_validate_batch,
                                                                  keep_prob: 1.0})
                    full_prediction.append(__y_pred * RESCALE)
                    actual_rul.append(__y * RESCALE)
                    error_list.append(error * RESCALE)
                    # x = np.reshape(x_validate_batch,[x_validate_batch.shape[0]*x_validate_batch.shape[1],x_validate_batch.shape[2]])
                    # y = np.reshape(x_train_batch,
                    #                [x_train_batch.shape[0] * x_train_batch.shape[1], x_train_batch.shape[2]])
                    # plt.figure(1)
                    # plt.plot(__y_pred, label="prediction")
                    # plt.plot(__y, label="expected")
                    # plt.plot(y,label="y")
                    # plt.figure(2)
                    # plt.plot(x,label="x")
                    # plt.legend()
                    # plt.show()
                full_prediction = np.array(full_prediction)
                full_prediction = full_prediction.ravel()
                actual_rul = np.array(actual_rul)
                actual_rul = actual_rul.ravel()
                error_list = np.array(error_list)
                error_list = error_list.ravel()
                rmse = np.sqrt(np.sum(np.square(error_list)) / len(error_list))  # RMSE

                print(y_validation.shape, full_prediction.shape, "RMSE:", rmse, "Score:", scoring_func(error_list))
                if plot:
                    plt.plot(full_prediction, label="prediction")
                    plt.plot(actual_rul, label="expected")
                    plt.legend()
                    plt.show()


if __name__ == "__main__":
    ""
    dataset = "cmapss"
    file = 1
    TRAIN = True
    analyse_Data(dataset=dataset, files=[file], plot=False, min_max=False)
    if TRAIN: data_augmentation(files=file,
                                low=[10, 35, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330],
                                high=[35, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350],
                                plot=False)
    from data_processing import RESCALE, test_engine_id, true_rul

    CNNLSTM(dataset=dataset, Train=TRAIN, trj_wise=False)
    # InceptionLSTM(dataset=dataset, Train=TRAIN, trj_wise=True)
