import hashlib, os, json
import numpy as np
import sklearn.model_selection as skcv

def hasher(msg):
    if not isinstance(msg, bytes):
        msg = str(msg).encode('utf-8')
    hasher = hashlib.md5()
    hasher.update(msg)
    return hasher.hexdigest()

def load_data(kernel, save_dir=''):
    '''Returns pre-loaded datasets & fields; Adapted from fx_data_load'''
    karoo_dir = 'karoo_gp'  #os.path.dirname(os.path.realpath(__file__))
    filename = {
        'c': karoo_dir + '/files/data_CLASSIFY.csv',
        'r': karoo_dir + '/files/data_REGRESS.csv',
        'm': karoo_dir + '/files/data_MATCH.csv',
        'p': karoo_dir + '/files/data_PLAY.csv',
    }[kernel]

    data_x = np.loadtxt(filename, skiprows=1, delimiter=',', dtype=float)
    data_x = data_x[:,0:-1]  # load all but the right-most column
    # load only right-most column (class labels)
    data_y = np.loadtxt(filename, skiprows=1, usecols=(-1,),
                        delimiter=',', dtype=float)
    header = open(filename, 'r')  # open file to be read (below)
    terminals = header.readline().split(',')
    terminals[-1] = terminals[-1].replace('\n', '')

    fitness_type = {'c': 'max', 'r': 'min', 'm': 'max', 'p': ''}[kernel]

    func_dict = {
        'c': karoo_dir + '/files/operators_CLASSIFY.csv',
        'r': karoo_dir + '/files/operators_REGRESS.csv',
        'm': karoo_dir + '/files/operators_MATCH.csv',
        'p': karoo_dir + '/files/operators_PLAY.csv',
    }
    functions = np.loadtxt(func_dict[kernel], delimiter=',',
                           skiprows=1, dtype=str)
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


    # generate a unique directory name
    runs_dir = os.path.join(os.getcwd(), 'runs')
    path = os.path.join(runs_dir, save_dir + '/')
    if not os.path.isdir(path):
        os.makedirs(path)  # make a unique directory

    savefile = {}  # a dictionary to hold .csv filenames
    for k in ['a', 'b', 'f', 's']:
        savefile[k] = f'{path}population_{k}.csv'
        open(savefile[k], 'w').close()

    return dict(data_train=data_train,
                class_labels=class_labels,
                data_train_rows=data_train_rows,
                savefile=savefile,
                terminals=terminals,
                functions=functions,
                fitness_type=fitness_type)

def dump_json(dictionary, path):
    with open(path, 'w') as f:
        json.dump(dictionary, f)
