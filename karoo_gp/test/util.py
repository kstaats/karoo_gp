import hashlib, json
import numpy as np
import pandas as pd

def hasher(msg):
    if not isinstance(msg, bytes):
        msg = str(msg).encode('utf-8')
    hasher = hashlib.md5()
    hasher.update(msg)
    return hasher.hexdigest()

def load_data(tmp_path, paths, kernel):
    """Return a dict with datasets and params for a given kernel and paths"""

    # Load functions
    func_path = paths.func_files[kernel]
    functions = np.loadtxt(func_path, delimiter=',', skiprows=1, dtype=str)
    functions = [f[0] for f in functions]  # Arity is now hard-coded by symbol

    # Load data, extract terminals, X and y
    data_path = paths.data_files[kernel]
    dataset = pd.read_csv(data_path)
    y = dataset.pop('s')
    terminals = list(dataset.keys())
    X, y = dataset.to_numpy(), y.to_numpy()

    # Create savefile names
    savefile = {}
    for k in ['a', 'b', 'f', 's']:
        savefile[k] = f'{tmp_path}population_{k}.csv'
        open(savefile[k], 'w').close()

    return dict(X=X,
                y=y,
                savefile=savefile,
                terminals=terminals,
                functions=functions)

def dump_json(dictionary, path):
    """Save a dictionary to a json file

    Helper function to debug or set expected output for tests
    """
    with open(path, 'w') as f:
        json.dump(dictionary, f)
