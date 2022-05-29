import pathlib
import subprocess

import pytest
import numpy as np


def parse_log(log_path, root_dir):
    """Extract dataset file and the results from log_test.txt."""
    text = log_path.read_text()
    # ignore the head and the date, get the dataset path and the rest
    head, date, dataset, rest = text.split('\n', 3)
    dataset = pathlib.Path(dataset.strip().split(': ', 1)[-1])
    if dataset.is_absolute():
        # the test data files use relative paths, make this relative too
        dataset = dataset.relative_to(root_dir)
    return dataset, rest


@pytest.mark.parametrize('seed', ['1000'])
@pytest.mark.parametrize('bas', ['3'])
@pytest.mark.parametrize('typ', ['f', 'g', 'r'])
@pytest.mark.parametrize('ker', ['c', 'r', 'm'])
def test_cli(tmp_path, paths, ker, typ, bas, seed):
    """Test that the CLI yields consistent results with different kernels/trees."""
    # default pop is 10, except for m-g and m-f that need higher pop to pass
    pop = str({('m', 'g'): 35, ('m', 'f'): 50}.get((ker, typ), 10))
    data_file = paths.data_files[ker]  # get the right data file for the kernel
    cmd = ['python3', paths.karoo, '-ker', ker, '-typ', typ, '-bas', bas,
           '-pop', pop, '-rsd', seed, '-fil', data_file]
    print(' '.join(map(str, cmd)))
    cp = subprocess.run(cmd, cwd=tmp_path)  # run Karoo in a tmp dir
    assert cp.returncode == 0  # check that the run was successful

    runs_dir = tmp_path / 'runs'  # runs are in the 'runs' dir
    runs = list(runs_dir.iterdir())
    assert len(runs) == 1  # there should be only one run

    log_test_rd = runs_dir / runs[0] / 'log_test.txt'
    log_test_td = paths.test_data / f'log_test[{ker}-{typ}].txt'
    actual_dataset, actual_log = parse_log(log_test_rd, paths.root)
    expected_dataset, expected_log = parse_log(log_test_td, paths.root)

    # check that both datasets match the input dataset
    assert actual_dataset == expected_dataset == data_file.relative_to(paths.root)
    assert actual_log == expected_log  # compare the content of the log_test
