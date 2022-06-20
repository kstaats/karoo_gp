import json
import pathlib
import platform
import subprocess

import pytest
import numpy as np


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

    # load the actual and expected jsons
    actual_res_path = runs_dir / runs[0] / 'results.json'
    expected_res_path = paths.test_data / f'results[{ker}-{typ}].json'
    with actual_res_path.open() as actual_res_file:
        actual_json = json.load(actual_res_file)
    with expected_res_path.open() as expected_res_file:
        expected_json = json.load(expected_res_file)

    # remove the date since it's different
    actual_json.pop('launched')
    expected_json.pop('launched')
    # extract the dataset path since it's different
    actual_dataset = pathlib.Path(actual_json.pop('dataset'))
    expected_dataset = pathlib.Path(expected_json.pop('dataset'))

    # check that both datasets match the input dataset
    assert (data_file.relative_to(paths.root) ==
            actual_dataset.relative_to(paths.root) ==
            expected_dataset)

    # Not needed after implementing sigfig_round in the Regression kernel
    # if ker == 'r' and typ == 'f' and platform.mac_ver()[-1] == 'arm64':
    #     # the Apple M1 seems to have some accuracy problem that leads
    #     # this test to fail, so extract the score and approximate it
    #     actual_score = actual_json.pop('score')
    #     expected_score = expected_json.pop('score')
    #     assert pytest.approx(actual_score) == expected_score

    # compare the content of the two jsons
    assert actual_json == expected_json
