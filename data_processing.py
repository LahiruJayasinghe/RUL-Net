import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

MAXLIFE = 120
SCALE = 1
RESCALE = 1
true_rul = []
test_engine_id = 0
training_engine_id = 0


def kink_RUL(cycle_list, max_cycle):
    '''
    Piecewise linear function with zero gradient and unit gradient

            ^
            |
    MAXLIFE |-----------
            |            \
            |             \
            |              \
            |               \
            |                \
            |----------------------->
    '''
    knee_point = max_cycle - MAXLIFE
    kink_RUL = []
    stable_life = MAXLIFE
    for i in range(0, len(cycle_list)):
        if i < knee_point:
            kink_RUL.append(MAXLIFE)
        else:
            tmp = kink_RUL[i - 1] - (stable_life / (max_cycle - knee_point))
            kink_RUL.append(tmp)

    return kink_RUL


def compute_rul_of_one_id(FD00X_of_one_id, max_cycle_rul=None):
    '''
    Enter the data of an engine_id of train_FD001 and output the corresponding RUL (remaining life) of these data.
    type is list
    '''

    cycle_list = FD00X_of_one_id['cycle'].tolist()
    if max_cycle_rul is None:
        max_cycle = max(cycle_list)  # Failure cycle
    else:
        max_cycle = max(cycle_list) + max_cycle_rul
        # print(max(cycle_list), max_cycle_rul)

    # return kink_RUL(cycle_list,max_cycle)
    return kink_RUL(cycle_list, max_cycle)


def compute_rul_of_one_file(FD00X, id='engine_id', RUL_FD00X=None):
    '''
    Input train_FD001, output a list
    '''
    rul = []
    # In the loop train, each id value of the 'engine_id' column
    if RUL_FD00X is None:
        for _id in set(FD00X[id]):
            rul.extend(compute_rul_of_one_id(FD00X[FD00X[id] == _id]))
        return rul
    else:
        rul = []
        for _id in set(FD00X[id]):
            # print("#### id ####", int(RUL_FD00X.iloc[_id - 1]))
            true_rul.append(int(RUL_FD00X.iloc[_id - 1]))
            rul.extend(compute_rul_of_one_id(FD00X[FD00X[id] == _id], int(RUL_FD00X.iloc[_id - 1])))
        return rul


def get_CMAPSSData(save=False, save_training_data=True, save_testing_data=True, files=[1, 2, 3, 4, 5],
                   min_max_norm=False):
    '''
    :param save: switch to load the already preprocessed data or begin preprocessing of raw data
    :param save_training_data: same functionality as 'save' but for training data only
    :param save_testing_data: same functionality as 'save' but for testing data only
    :param files: to indicate which sub dataset needed to be loaded for operations
    :param min_max_norm: switch to enable min-max normalization
    :return: function will save the preprocessed training and testing data as numpy objects
    '''

    if save == False:
        return np.load("normalized_train_data.npy"), np.load("normalized_test_data.npy"), pd.read_csv(
            'normalized_train_data.csv', index_col=[0]), pd.read_csv('normalized_test_data.csv', index_col=[0])

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    if save_training_data:  ### Training ###

        train_FD001 = pd.read_table("./CMAPSSData/train_FD001.txt", header=None, delim_whitespace=True)
        train_FD002 = pd.read_table("./CMAPSSData/train_FD002.txt", header=None, delim_whitespace=True)
        train_FD003 = pd.read_table("./CMAPSSData/train_FD003.txt", header=None, delim_whitespace=True)
        train_FD004 = pd.read_table("./CMAPSSData/train_FD004.txt", header=None, delim_whitespace=True)
        train_FD001.columns = column_name
        train_FD002.columns = column_name
        train_FD003.columns = column_name
        train_FD004.columns = column_name

        previous_len = 0
        frames = []
        for data_file in ['train_FD00' + str(i) for i in files]:  # load subdataset by subdataset

            #### standard normalization ####
            mean = eval(data_file).iloc[:, 2:len(list(eval(data_file)))].mean()
            std = eval(data_file).iloc[:, 2:len(list(eval(data_file)))].std()
            std.replace(0, 1, inplace=True)
            # print("std", std)
            ################################

            if min_max_norm:
                scaler = MinMaxScaler()
                eval(data_file).iloc[:, 2:len(list(eval(data_file)))] = scaler.fit_transform(
                    eval(data_file).iloc[:, 2:len(list(eval(data_file)))])
            else:
                eval(data_file).iloc[:, 2:len(list(eval(data_file)))] = (eval(data_file).iloc[:, 2:len(
                    list(eval(data_file)))] - mean) / std

            eval(data_file)['RUL'] = compute_rul_of_one_file(eval(data_file))
            current_len = len(eval(data_file))
            # print(eval(data_file).index)
            eval(data_file).index = range(previous_len, previous_len + current_len)
            previous_len = previous_len + current_len
            # print(eval(data_file).index)
            frames.append(eval(data_file))
            print(data_file)

        train = pd.concat(frames)
        global training_engine_id
        training_engine_id = train['engine_id']
        train = train.drop('engine_id', 1)
        train = train.drop('cycle', 1)
        # if files[0] == 1 or files[0] == 3:
        #     train = train.drop('setting3', 1)
        #     train = train.drop('s18', 1)
        #     train = train.drop('s19', 1)

        train_values = train.values * SCALE
        np.save('normalized_train_data.npy', train_values)
        train.to_csv('normalized_train_data.csv')
        ###########
    else:
        train = pd.read_csv('normalized_train_data.csv', index_col=[0])
        train_values = train.values

    if save_testing_data:  ### testing ###

        test_FD001 = pd.read_table("./CMAPSSData/test_FD001.txt", header=None, delim_whitespace=True)
        test_FD002 = pd.read_table("./CMAPSSData/test_FD002.txt", header=None, delim_whitespace=True)
        test_FD003 = pd.read_table("./CMAPSSData/test_FD003.txt", header=None, delim_whitespace=True)
        test_FD004 = pd.read_table("./CMAPSSData/test_FD004.txt", header=None, delim_whitespace=True)
        test_FD001.columns = column_name
        test_FD002.columns = column_name
        test_FD003.columns = column_name
        test_FD004.columns = column_name

        # load RUL data
        RUL_FD001 = pd.read_table("./CMAPSSData/RUL_FD001.txt", header=None, delim_whitespace=True)
        RUL_FD002 = pd.read_table("./CMAPSSData/RUL_FD002.txt", header=None, delim_whitespace=True)
        RUL_FD003 = pd.read_table("./CMAPSSData/RUL_FD003.txt", header=None, delim_whitespace=True)
        RUL_FD004 = pd.read_table("./CMAPSSData/RUL_FD004.txt", header=None, delim_whitespace=True)
        RUL_FD001.columns = ['RUL']
        RUL_FD002.columns = ['RUL']
        RUL_FD003.columns = ['RUL']
        RUL_FD004.columns = ['RUL']

        previous_len = 0
        frames = []
        for (data_file, rul_file) in [('test_FD00' + str(i), 'RUL_FD00' + str(i)) for i in files]:
            mean = eval(data_file).iloc[:, 2:len(list(eval(data_file)))].mean()
            std = eval(data_file).iloc[:, 2:len(list(eval(data_file)))].std()
            std.replace(0, 1, inplace=True)

            if min_max_norm:
                scaler = MinMaxScaler()
                eval(data_file).iloc[:, 2:len(list(eval(data_file)))] = scaler.fit_transform(
                    eval(data_file).iloc[:, 2:len(list(eval(data_file)))])
            else:
                eval(data_file).iloc[:, 2:len(list(eval(data_file)))] = (eval(data_file).iloc[:, 2:len(
                    list(eval(data_file)))] - mean) / std

            eval(data_file)['RUL'] = compute_rul_of_one_file(eval(data_file), RUL_FD00X=eval(rul_file))
            current_len = len(eval(data_file))
            eval(data_file).index = range(previous_len, previous_len + current_len)
            previous_len = previous_len + current_len
            frames.append(eval(data_file))
            print(data_file)
            if len(files) == 1:
                global test_engine_id
                test_engine_id = eval(data_file)['engine_id']

        test = pd.concat(frames)
        test = test.drop('engine_id', 1)
        test = test.drop('cycle', 1)
        # if files[0] == 1 or files[0] == 3:
        #     test = test.drop('setting3', 1)
        #     test = test.drop('s18', 1)
        #     test = test.drop('s19', 1)

        test_values = test.values * SCALE
        np.save('normalized_test_data.npy', test_values)
        test.to_csv('normalized_test_data.csv')
        ###########
    else:
        test = pd.read_csv('normalized_test_data.csv', index_col=[0])
        test_values = test.values

    return train_values, test_values, train, test


def get_PHM08Data(save=False):
    """
    Function is to load PHM 2008 challenge dataset

    """

    if save == False:
        return np.load("./PHM08/processed_data/phm_training_data.npy"), np.load("./PHM08/processed_data/phm_testing_data.npy"), np.load(
            "./PHM08/processed_data/phm_original_testing_data.npy")

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']
    phm_training_data = pd.read_table("./PHM08/train.txt", header=None, delim_whitespace=True)
    phm_training_data.columns = column_name
    phm_testing_data = pd.read_table("./PHM08/final_test.txt", header=None, delim_whitespace=True)
    phm_testing_data.columns = column_name

    print("phm training")
    mean = phm_training_data.iloc[:, 2:len(list(phm_training_data))].mean()
    std = phm_training_data.iloc[:, 2:len(list(phm_training_data))].std()
    phm_training_data.iloc[:, 2:len(list(phm_training_data))] = (phm_training_data.iloc[:, 2:len(
        list(phm_training_data))] - mean) / std
    phm_training_data['RUL'] = compute_rul_of_one_file(phm_training_data)

    print("phm testing")
    mean = phm_testing_data.iloc[:, 2:len(list(phm_testing_data))].mean()
    std = phm_testing_data.iloc[:, 2:len(list(phm_testing_data))].std()
    phm_testing_data.iloc[:, 2:len(list(phm_testing_data))] = (phm_testing_data.iloc[:, 2:len(
        list(phm_testing_data))] - mean) / std
    phm_testing_data['RUL'] = 0
    #phm_testing_data['RUL'] = compute_rul_of_one_file(phm_testing_data)

    train_engine_id = phm_training_data['engine_id']
    # print(phm_training_engine_id[phm_training_engine_id==1].index)
    phm_training_data = phm_training_data.drop('engine_id', 1)
    phm_training_data = phm_training_data.drop('cycle', 1)

    global test_engine_id
    test_engine_id = phm_testing_data['engine_id']
    phm_testing_data = phm_testing_data.drop('engine_id', 1)
    phm_testing_data = phm_testing_data.drop('cycle', 1)

    phm_training_data = phm_training_data.values
    phm_testing_data = phm_testing_data.values

    engine_ids = train_engine_id.unique()
    train_test_split = np.random.rand(len(engine_ids)) < 0.80
    train_engine_ids = engine_ids[train_test_split]
    test_engine_ids = engine_ids[~train_test_split]

    # test_engine_id = pd.Series(test_engine_ids)


    training_data = phm_training_data[train_engine_id[train_engine_id == train_engine_ids[0]].index]
    for id in train_engine_ids[1:]:
        tmp = phm_training_data[train_engine_id[train_engine_id == id].index]
        training_data = np.concatenate((training_data, tmp))
    # print(training_data.shape)

    testing_data = phm_training_data[train_engine_id[train_engine_id == test_engine_ids[0]].index]
    for id in test_engine_ids[1:]:
        tmp = phm_training_data[train_engine_id[train_engine_id == id].index]
        testing_data = np.concatenate((testing_data, tmp))
    # print(testing_data.shape)

    print(phm_training_data.shape, phm_testing_data.shape, training_data.shape, testing_data.shape)

    np.save("./PHM08/processed_data/phm_training_data.npy", training_data)
    np.savetxt("./PHM08/processed_data/phm_training_data.txt", training_data, delimiter=" ")
    np.save("./PHM08/processed_data/phm_testing_data.npy", testing_data)
    np.savetxt("./PHM08/processed_data/phm_testing_data.txt", testing_data, delimiter=" ")
    np.save("./PHM08/processed_data/phm_original_testing_data.npy", phm_testing_data)
    np.savetxt("./PHM08/processed_data/phm_original_testing_data.csv", phm_testing_data, delimiter=",")

    return training_data, testing_data, phm_testing_data


def data_augmentation(files=1, low=[10, 40, 90, 170], high=[35, 85, 160, 250], plot=False, combine=False):
    '''
    This helper function only augments the training data to look like testing data.
    Training data always run to a failure. But testing data is mostly stop before a failure.
    Therefore, training data augmented to have scenarios without failure

    :param files: select wich sub CMPASS dataset
    :param low: lower bound for the random selection of the engine cycle
    :param high: upper bound for the random selection of the engine cycle
    :param plot: switch to plot the augmented data
    :return:
    '''

    DEBUG = False

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    ### Loading original data ###
    if files == "phm":
        train_FD00x = pd.read_table("./PHM08/processed_data/phm_training_data.txt", header=None, delim_whitespace=True)
        train_FD00x.drop(train_FD00x.columns[len(train_FD00x.columns) - 1], axis=1, inplace=True)
        train_FD00x.columns = column_name
    else:
        if combine:
            train_FD00x,_,_ = combine_FD001_and_FD003()
        else:
            file_path = "./CMAPSSData/train_FD00" + str(files) + ".txt"
            train_FD00x = pd.read_table(file_path, header=None, delim_whitespace=True)
            train_FD00x.columns = column_name
            print(file_path.split("/")[-1])

        ### Standered Normal ###
        mean = train_FD00x.iloc[:, 2:len(list(train_FD00x))].mean()
        std = train_FD00x.iloc[:, 2:len(list(train_FD00x))].std()
        std.replace(0, 1, inplace=True)
        train_FD00x.iloc[:, 2:len(list(train_FD00x))] = (train_FD00x.iloc[:, 2:len(list(train_FD00x))] - mean) / std

    final_train_FD = train_FD00x.copy()
    previous_len = 0
    frames = []
    for i in range(len(high)):
        train_FD = train_FD00x.copy()
        train_engine_id = train_FD['engine_id']
        engine_ids = train_engine_id.unique()
        total_ids = len(engine_ids)
        train_rul = []
        print("*************", final_train_FD.shape, total_ids, low[i], high[i], "*****************")

        for id in range(1, total_ids + 1):

            train_engine_id = train_FD['engine_id']
            indexes = train_engine_id[train_engine_id == id].index  ### filter indexes related to id
            traj_data = train_FD.loc[indexes]  ### filter trajectory data

            cutoff_cycle = random.randint(low[i], high[i])  ### randomly selecting the cutoff point of the engine cycle

            if cutoff_cycle > max(traj_data['cycle']):
                cutoff_cycle = max(traj_data['cycle'])

            train_rul.append(max(traj_data['cycle']) - cutoff_cycle)  ### collecting remaining cycles

            cutoff_cycle_index = traj_data['cycle'][traj_data['cycle'] == cutoff_cycle].index  ### cutoff cycle index

            if DEBUG:
                print("traj_shape: ", traj_data.shape, "current_engine_id:", id, "cutoff_cycle:", cutoff_cycle,
                      "cutoff_index", cutoff_cycle_index, "engine_fist_index", indexes[0], "engine_last_index",
                      indexes[-1])

            ### removing rows after cutoff cycle index ###
            if cutoff_cycle_index[0] != indexes[-1]:
                drop_range = list(range(cutoff_cycle_index[0] + 1, indexes[-1] + 1))
                train_FD.drop(train_FD.index[drop_range], inplace=True)
                train_FD.reset_index(drop=True, inplace=True)

        ### calculating the RUL for augmented data
        train_rul = pd.DataFrame.from_dict({'RUL': train_rul})
        train_FD['RUL'] = compute_rul_of_one_file(train_FD, RUL_FD00X=train_rul)

        ### changing the engine_id for augmented data
        train_engine_id = train_FD['engine_id']
        for id in range(1, total_ids + 1):
            indexes = train_engine_id[train_engine_id == id].index
            train_FD.loc[indexes, 'engine_id'] = id + total_ids * (i + 1)

        if i == 0:  # should only execute at the first iteration
            final_train_FD['RUL'] = compute_rul_of_one_file(final_train_FD)
            current_len = len(final_train_FD)
            final_train_FD.index = range(previous_len, previous_len + current_len)
            previous_len = previous_len + current_len

        ### Re-indexing the augmented data
        train_FD['RUL'].index = range(previous_len, previous_len + len(train_FD))
        previous_len = previous_len + len(train_FD)

        final_train_FD = pd.concat(
            [final_train_FD, train_FD])  # concatanete the newly augmented data with previous data

    frames.append(final_train_FD)
    train = pd.concat(frames)
    train.reset_index(drop=True, inplace=True)

    train_engine_id = train['engine_id']
    # print(train_engine_id)
    engine_ids = train_engine_id.unique()
    # print(engine_ids[1:])
    np.random.shuffle(engine_ids)
    # print(engine_ids)

    training_data = train.loc[train_engine_id[train_engine_id == engine_ids[0]].index]
    training_data.reset_index(drop=True, inplace=True)
    previous_len = len(training_data)
    for id in engine_ids[1:]:
        traj_data = train.loc[train_engine_id[train_engine_id == id].index]
        current_len = len(traj_data)
        traj_data.index = range(previous_len, previous_len + current_len)
        previous_len = previous_len + current_len
        training_data = pd.concat([training_data, traj_data])


    global training_engine_id
    training_engine_id = training_data['engine_id']

    training_data = training_data.drop('engine_id', 1)
    training_data = training_data.drop('cycle', 1)
    # if files == 1 or files == 3:
    #     training_data = training_data.drop('setting3', 1)
    #     training_data = training_data.drop('s18', 1)
    #     training_data = training_data.drop('s19', 1)

    training_data_values = training_data.values * SCALE
    np.save('normalized_train_data.npy', training_data_values)
    training_data.to_csv('normalized_train_data.csv')


    train = training_data_values
    x_train = train[:, :train.shape[1] - 1]
    y_train = train[:, train.shape[1] - 1] * RESCALE
    print("training in augmentation", x_train.shape, y_train.shape)

    if plot:
        plt.plot(y_train, label="train")

        plt.figure()
        plt.plot(x_train)
        plt.title("train")
        # plt.figure()
        # plt.plot(y_train)
        # plt.title("test")

        plt.show()


def analyse_Data(dataset, files=None, plot=True, min_max=False):
    '''
    Generate pre-processed data according to the given dataset
    :param dataset: choose between "phm" for PHM 2008 dataset or "cmapss" for CMAPSS data set with file number
    :param files: Only for CMAPSS dataset to select sub dataset
    :param min_max: switch to allow min-max normalization
    :return:
    '''

    if dataset == "phm":
        training_data, testing_data, phm_testing_data = get_PHM08Data(save=True)

        x_phmtrain = training_data[:, :training_data.shape[1] - 1]
        y_phmtrain = training_data[:, training_data.shape[1] - 1]

        x_phmtest = testing_data[:, :testing_data.shape[1] - 1]
        y_phmtest = testing_data[:, testing_data.shape[1] - 1]

        print("phmtrain", x_phmtrain.shape, y_phmtrain.shape)

        print("phmtest", x_phmtrain.shape, y_phmtrain.shape)
        print("phmtest", phm_testing_data.shape)

        if plot:
            # plt.plot(x_phmtrain, label="phmtrain_x")
            plt.figure()
            plt.plot(y_phmtrain, label="phmtrain_y")

            # plt.figure()
            # plt.plot(x_phmtest, label="phmtest_x")
            plt.figure()
            plt.plot(y_phmtest, label="phmtest_y")

            # plt.figure()
            # plt.plot(phm_testing_data, label="test")
            plt.show()

    elif dataset == "cmapss":
        training_data, testing_data, training_pd, testing_pd = get_CMAPSSData(save=True, files=files,
                                                                              min_max_norm=min_max)
        x_train = training_data[:, :training_data.shape[1] - 1]
        y_train = training_data[:, training_data.shape[1] - 1]
        print("training", x_train.shape, y_train.shape)

        x_test = testing_data[:, :testing_data.shape[1] - 1]
        y_test = testing_data[:, testing_data.shape[1] - 1]
        print("testing", x_test.shape, y_test.shape)

        if plot:
            plt.plot(y_train, label="train")
            plt.figure()
            plt.plot(y_test, label="test")

            plt.figure()
            plt.plot(x_train)
            plt.title("train: FD00" + str(files[0]))
            plt.figure()
            plt.plot(y_train)
            plt.title("train: FD00" + str(files[0]))
            plt.show()


def combine_FD001_and_FD003():
    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                   's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                   's15', 's16', 's17', 's18', 's19', 's20', 's21']

    train_FD001 = pd.read_table("./CMAPSSData/train_FD001.txt", header=None, delim_whitespace=True)
    train_FD003 = pd.read_table("./CMAPSSData/train_FD003.txt", header=None, delim_whitespace=True)
    train_FD001.columns = column_name
    train_FD003.columns = column_name

    FD001_max_engine_id = max(train_FD001['engine_id'])
    train_FD003['engine_id'] = train_FD003['engine_id'] + FD001_max_engine_id
    train_FD003.index = range(len(train_FD001), len(train_FD001) + len(train_FD003))
    train_FD001_FD002 = pd.concat([train_FD001,train_FD003])

    test_FD001 = pd.read_table("./CMAPSSData/test_FD001.txt", header=None, delim_whitespace=True)
    test_FD003 = pd.read_table("./CMAPSSData/test_FD003.txt", header=None, delim_whitespace=True)
    test_FD001.columns = column_name
    test_FD003.columns = column_name

    FD001_max_engine_id = max(test_FD001['engine_id'])
    test_FD003['engine_id'] = test_FD003['engine_id'] + FD001_max_engine_id
    test_FD003.index = range(len(test_FD001), len(test_FD001) + len(test_FD003))
    test_FD001_FD002 = pd.concat([test_FD001,test_FD003])

    RUL_FD001 = pd.read_table("./CMAPSSData/RUL_FD001.txt", header=None, delim_whitespace=True)
    RUL_FD003 = pd.read_table("./CMAPSSData/RUL_FD003.txt", header=None, delim_whitespace=True)
    RUL_FD001.columns = ['RUL']
    RUL_FD003.columns = ['RUL']
    RUL_FD003.index = range(len(RUL_FD001), len(RUL_FD001) + len(RUL_FD003))
    RUL_FD001_FD002 = pd.concat([test_FD001, test_FD003])

    return train_FD001_FD002,test_FD001_FD002,RUL_FD001_FD002
