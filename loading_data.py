"""Loading data sets"""
import pandas as pd
import numpy as np

def add_rul_1(df):
    """

    :param df: raw data frame
    :return: data frame labeled with targets
    """
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (piece-wise Linear)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]

    result_frame["RUL"] = remaining_useful_life
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def load_FD001(cut):
    """

    :param cut: upper limit for target RULs
    :return: grouped data per sample
    """
    # load data FD001.py
    # define filepath to read data
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features, derived from EDA
    # drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors

    train.drop(labels=drop_labels, axis=1, inplace=True)
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())  # min-max normalization
    # data_norm = (data-data.mean())/data.std()  # standard normalization (optional)
    train_norm = pd.concat([title, data_norm], axis=1)
    train_norm = add_rul_1(train_norm)
    # as in piece-wise linear function, there is an upper limit for target RUL,
    # however, experimental results shows this goes even better without it:
    # train_norm['RUL'].clip(upper=cut, inplace=True)
    group = train_norm.groupby(by="unit_nr")

    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    test_norm = pd.concat([title, data_norm], axis=1)
    group_test = test_norm.groupby(by="unit_nr")

    return group, group_test, y_test



def get_norm_data(train, test):
    """

    :param cut: upper limit for target RULs
    :return: grouped data per sample
    """

    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())  # min-max normalization
    # data_norm = (data-data.mean())/data.std()  # standard normalization (optional)
    train_norm = pd.concat([title, data_norm], axis=1)
    # train_norm = add_rul_1(train_norm)
    # rrul is added as the last column
    # as in piece-wise linear function, there is an upper limit for target RUL,
    # however, experimental results shows this goes even better without it:
    # train_norm['RUL'].clip(upper=cut, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    test_norm = pd.concat([title, data_norm], axis=1)



    return train_norm, test_norm




def load_data_FD001_CNN():
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_raw = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test_raw = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL']).to_numpy()

    grouped_by_unit = train_raw.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max().to_numpy()

    grouped_by_unit_t = test_raw.groupby(by="unit_nr")
    max_cycle_t = grouped_by_unit_t["time_cycles"].max().to_numpy()

    # drop non-informative features, derived from EDA
    # drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train_raw.drop(labels=drop_labels, axis=1, inplace=True)
    test_raw.drop(labels=drop_labels, axis=1, inplace=True)

    remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
                         's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

    return train_raw, test_raw, max_cycle, max_cycle_t, y_test


def load_FD002(cut):
    """

    :param cut: upper limit for target RULs
    :return: grouped data per sample
    """
    # load data FD001.py
    # define filepath to read data
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD002.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD002.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD002.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features, derived from EDA
    # drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors

    train.drop(labels=drop_labels, axis=1, inplace=True)
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())  # min-max normalization
    # data_norm = (data-data.mean())/data.std()  # standard normalization (optional)
    train_norm = pd.concat([title, data_norm], axis=1)
    train_norm = add_rul_1(train_norm)
    # as in piece-wise linear function, there is an upper limit for target RUL,
    # however, experimental results shows this goes even better without it:
    # train_norm['RUL'].clip(upper=cut, inplace=True)
    group = train_norm.groupby(by="unit_nr")

    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    test_norm = pd.concat([title, data_norm], axis=1)
    group_test = test_norm.groupby(by="unit_nr")

    return group, group_test, y_test


def load_data_FD003():
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_raw = pd.read_csv((dir_path + 'train_FD003.txt'), sep='\s+', header=None, names=col_names)
    test_raw = pd.read_csv((dir_path + 'test_FD003.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD003.txt'), sep='\s+', header=None, names=['RUL']).to_numpy()

    grouped_by_unit = train_raw.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max().to_numpy()

    grouped_by_unit_t = test_raw.groupby(by="unit_nr")
    max_cycle_t = grouped_by_unit_t["time_cycles"].max().to_numpy()

    # drop non-informative features, derived from EDA
    drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train_raw.drop(labels=drop_labels, axis=1, inplace=True)
    test_raw.drop(labels=drop_labels, axis=1, inplace=True)

    return train_raw, test_raw, max_cycle, max_cycle_t, y_test


def load_data_FD004():
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train_raw = pd.read_csv((dir_path + 'train_FD004.txt'), sep='\s+', header=None, names=col_names)
    test_raw = pd.read_csv((dir_path + 'test_FD004.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD004.txt'), sep='\s+', header=None, names=['RUL']).to_numpy()

    grouped_by_unit = train_raw.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max().to_numpy()

    grouped_by_unit_t = test_raw.groupby(by="unit_nr")
    max_cycle_t = grouped_by_unit_t["time_cycles"].max().to_numpy()

    # drop non-informative features, derived from EDA
    # drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names
    train_raw.drop(labels=drop_labels, axis=1, inplace=True)
    test_raw.drop(labels=drop_labels, axis=1, inplace=True)

    return train_raw, test_raw, max_cycle, max_cycle_t, y_test


def get_info(train_raw, test_raw):
    mm = MinMaxScaler()
    ss = StandardScaler()

    X = train_raw.iloc[:, 2:] # (26031, 15)
    idx = train_raw.iloc[:, 0:2].to_numpy()
    X_ss = ss.fit_transform(X)

    X_t = test_raw.iloc[:, 2:] # (13096, 15)
    idx_t = test_raw.iloc[:, 0:2].to_numpy()
    Xt_ss = ss.fit_transform(X_t)

    nf = X_ss.shape[1] # 15
    ns = X_ss.shape[0]  # 20361
    ns_t = Xt_ss.shape[0]  # 13096

    return X_ss, idx, Xt_ss, idx_t, nf, ns, ns_t
