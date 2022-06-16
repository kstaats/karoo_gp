import hashlib, os, json
import numpy as np
import sklearn.model_selection as skcv

def hasher(msg):
    if not isinstance(msg, bytes):
        msg = str(msg).encode('utf-8')
    hasher = hashlib.md5()
    hasher.update(msg)
    return hasher.hexdigest()

def load_data(tmp_path, paths, kernel):
    """Return a dict with datasets and params for a given kernel and paths"""
    data_file = paths.data_files[kernel]
    data_x = np.loadtxt(data_file, skiprows=1, delimiter=',', dtype=float)
    data_x = data_x[:,0:-1]  # load all but the right-most column
    # load only right-most column (class labels)
    data_y = np.loadtxt(data_file, skiprows=1, usecols=(-1,),
                        delimiter=',', dtype=float)
    header = open(data_file, 'r')  # open file to be read (below)
    terminals = header.readline().split(',')
    terminals[-1] = terminals[-1].replace('\n', '')
    fitness_type = {'c': 'max', 'r': 'min', 'm': 'max', 'p': ''}[kernel]
    func_file = paths.func_files[kernel]
    functions = np.loadtxt(func_file, delimiter=',', skiprows=1, dtype=str)
    class_labels = len(np.unique(data_y))
    if len(data_x) < 11:
        data_train = np.c_[data_x, data_y]
        data_test = np.c_[data_x, data_y]
    else:
        x_train, x_test, y_train, y_test = skcv.train_test_split(
            data_x, data_y, test_size=0.2
        )  # 80/20 TRAIN/TEST split
        data_train = np.c_[x_train, y_train]
        data_test = np.c_[x_test, y_test]

    # data_train_cols = len(data_train[0,:])  # qty count
    data_train_rows = len(data_train[:,0])  # qty count
    # data_test_cols = len(data_test[0,:])  # qty count
    # data_test_rows = len(data_test[:,0])  # qty count

    savefile = {}  # a dictionary to hold .csv filenames
    for k in ['a', 'b', 'f', 's']:
        savefile[k] = f'{tmp_path}population_{k}.csv'
        open(savefile[k], 'w').close()

    return dict(data_train=data_train,
                data_test=data_test,
                class_labels=class_labels,
                data_train_rows=data_train_rows,
                savefile=savefile,
                terminals=terminals,
                functions=functions,
                fitness_type=fitness_type)

def dump_json(dictionary, path):
    with open(path, 'w') as f:
        json.dump(dictionary, f)
