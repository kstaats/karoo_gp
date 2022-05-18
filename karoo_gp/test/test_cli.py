import pathlib
import subprocess

import pytest
import numpy as np


@pytest.fixture
def root_dir():
    return pathlib.Path(__file__).resolve().parents[2]

@pytest.fixture
def karoo(root_dir):
    return root_dir / 'karoo-gp.py'

@pytest.fixture
def data_classify(root_dir):
    return root_dir / 'karoo_gp' / 'files' / 'data_CLASSIFY.csv'


@pytest.mark.parametrize('ker', ['c', 'r', 'm'])
@pytest.mark.parametrize('typ', ['f', 'g', 'r'])
def test_cli(tmp_path, karoo, data_classify, ker, typ):
    cmd = ['python3', karoo, '-ker', ker, '-typ', typ, '-bas', '3',
           '-pop', '10', '-fil', data_classify]
    cp = subprocess.run(cmd, cwd=tmp_path)
    assert cp.returncode == 0

    runs_dir = tmp_path / 'runs'  # runs are in the 'runs' dir
    runs = list(runs_dir.iterdir())
    assert len(runs) == 1  # there should be only one run

    # read the content of the log of the only run available
    log = (runs_dir / runs[0] / 'log_test.txt').read_text()

    # check that content of log matches what we expect
    extinct = 'your species has gone extinct!'
    expected = {
        ('f', 'c'): 'pl**2 - pl*sw + pw - sl - sw',
        ('f', 'r'): '2*pl - pw - sl + sw/pl - 1/pl',
        ('f', 'm'): extinct,
        ('g', 'c'): 'pl + pw - sl',
        ('g', 'r'): 'Tree 2 is the most fit, with expression:\n\n 1',
        ('g', 'm'): extinct,
        ('r', 'c'): 'pw*sw - sw',
        ('r', 'r'): 'Tree 10 is the most fit, with expression:\n\n 0',
        ('r', 'm'): extinct,
    }
    assert expected[(typ, ker)] in log
