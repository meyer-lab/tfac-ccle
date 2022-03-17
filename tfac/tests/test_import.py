import pytest
import numpy as np
from ..dataHelpers import import_LINCS_MEMA


@pytest.mark.parametrize("file", ["mcf10a_ssc_Level4.tsv.xz", "hmec240l_ssc_Level4.tsv.xz", "hmec122l_ssc_Level4.tsv.xz"])
def test_import(file):
    """ Test the data import. """
    tensor = import_LINCS_MEMA(file)
    assert np.all(np.isfinite(tensor))  # Should have no missingness
