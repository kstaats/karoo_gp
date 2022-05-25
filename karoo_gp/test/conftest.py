import pathlib

from collections import namedtuple

import pytest


@pytest.fixture
def paths():
    """Fixture that return an object with different paths as attributes."""
    fields = ['root', 'karoo', 'karoo_gp',
              'files', 'data_files', 'func_files',
              'test', 'test_data']
    Paths = namedtuple('Paths', fields)
    root = pathlib.Path(__file__).resolve().parents[2]
    files = root / 'karoo_gp' / 'files'
    # these two will be dicts of paths, not just individual paths
    data_files = {
        'c': files / 'data_CLASSIFY.csv',
        'r': files / 'data_REGRESS.csv',
        'm': files / 'data_MATCH.csv',
        'p': files / 'data_PLAY.csv',
    }
    func_files = {
        'c': files / 'operators_CLASSIFY.csv',
        'r': files / 'operators_REGRESS.csv',
        'm': files / 'operators_MATCH.csv',
        'p': files / 'operators_PLAY.csv',
    }
    return Paths(
        root = root,
        karoo = root / 'karoo-gp.py',
        karoo_gp = root / 'karoo_gp',
        files = files,
        data_files = data_files,
        func_files = func_files,
        test = root / 'karoo_gp' / 'test',
        test_data = root / 'karoo_gp' / 'test' / 'data',
    )
