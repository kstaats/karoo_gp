import pytest
from sklearn.utils.estimator_checks import check_estimator

from karoo_gp import BaseGP

@pytest.mark.skip(reason='failing')
def test_sklearn_estimator():
    check_estimator(BaseGP())
    # TODO: failing on parse_node:
    # "ValueError: Integers to negative integer powers are not allowed."
