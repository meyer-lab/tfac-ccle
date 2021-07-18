import tensorly as tl
from ..CMTF import perform_CMTF
from ..dataHelpers import form_tensor


def test_CMTF():
    """ Test the CMTF method with fake data. """
    tensor, rTensor, _, _ = form_tensor()
    tFac = perform_CMTF(tensor, rTensor)

    assert isinstance(tFac, tl.cp_tensor.CPTensor)
