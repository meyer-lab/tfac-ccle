"""
Tensor decomposition methods
"""
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, grad
from jax.config import config
from scipy.optimize import minimize
import tensorly as tl
from tensorly.decomposition._cp import initialize_cp


tl.set_backend('numpy')
config.update("jax_enable_x64", True)


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop = 0.0
    vBottom = 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        vTop += np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
        vBottom += np.sum(np.square(np.nan_to_num(tIn)))
    if mIn is not None:
        mMask = np.isfinite(mIn)
        vTop += np.sum(np.square(tl.cp_to_tensor(tFac.mFactor) * mMask - np.nan_to_num(mIn)))
        vBottom += np.sum(np.square(np.nan_to_num(mIn)))

    return 1.0 - vTop / vBottom


def cp_to_vec(tFac):
    return np.concatenate([tFac.factors[i].flatten() for i in range(2)])


def buildTensors(pIn, tensor, matrix, r, cost=False):
    """ Use parameter vector to build kruskal tensors. """
    assert tensor.shape[0] == matrix.shape[0]
    nN = np.cumsum(np.array(tensor.shape) * r)
    A = jnp.reshape(pIn[:nN[0]], (tensor.shape[0], r))
    B = jnp.reshape(pIn[nN[0]:nN[1]], (tensor.shape[1], r))

    kr = tl.tenalg.khatri_rao([A, B])
    unfold = tl.unfold(tensor, 2)
    unfoldM = tl.unfold(matrix, 2)

    if cost:
        cost = jnp.sum(jnp.linalg.lstsq(kr, unfold.T, rcond=None)[1])
        cost += jnp.sum(jnp.linalg.lstsq(kr, unfoldM.T, rcond=None)[1])
        return cost

    C = np.linalg.lstsq(kr, unfold.T, rcond=None)[0].T
    tFac = tl.cp_tensor.CPTensor((None, [A, B, C]))
    D = np.linalg.lstsq(kr, unfoldM.T, rcond=None)[0].T
    tFac.mFactor = tl.cp_tensor.CPTensor((None, [A, B, D]))
    return tFac


def cost(pIn, tOrig, mOrig, r):
    return buildTensors(pIn, tOrig, mOrig, r, cost=True)


def perform_CMTF(tOrig: np.ndarray, mOrig: np.ndarray, r=10):
    """ Perform CMTF decomposition by direct optimization. """
    # Checks
    tOrig = np.array(tOrig, dtype=float, order="C")
    mOrig = np.array(mOrig, dtype=float, order="C")
    assert tOrig.ndim == 3
    assert mOrig.ndim == 3
    assert tOrig.shape[0] == mOrig.shape[0]
    assert tOrig.shape[1] == mOrig.shape[1]

    tFac = initialize_cp(np.nan_to_num(tOrig), r)
    x0 = cp_to_vec(tFac)

    gF = value_and_grad(cost, 0)

    def gradF(x):
        a, b = gF(x, tOrig, mOrig, r)
        return a, np.array(b)

    def hvp(x, v):
        return grad(lambda x: jnp.vdot(gF(x, tOrig, mOrig, r)[1], v))(x)

    tl.set_backend('jax')
    res = minimize(gradF, x0, method="L-BFGS-B", jac=True)
    res = minimize(gradF, res.x, method="trust-constr", jac=True, hessp=hvp, options={"maxiter": 20})
    tl.set_backend('numpy')

    tFac = buildTensors(res.x, tOrig, mOrig, r)
    tFac.R2X = calcR2X(tFac, tOrig, mOrig)
    print(tFac.R2X)

    return tFac
