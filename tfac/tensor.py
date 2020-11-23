"""
Tensor decomposition methods
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker, parafac2
from tensorly.parafac2_tensor import parafac2_to_slice, apply_parafac2_projections
from tensorly.metrics.regression import variance as tl_var
from tensorly.tenalg import mode_dot


tl.set_backend("numpy")  # Set the backend


def z_score_values(A, cell_dim):
    """ Function that takes in the values tensor and z-scores it. """
    assert cell_dim < tl.ndim(A)
    convAxes = tuple([i for i in range(tl.ndim(A)) if i != cell_dim])
    convIDX = [None] * tl.ndim(A)
    convIDX[cell_dim] = slice(None)

    sigma = tl.tensor(np.std(tl.to_numpy(A), axis=convAxes))
    return A / sigma[tuple(convIDX)]


def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. """
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


def reorient_factors(factors):
    """ Reorient factors based on the sign of the mean so that only the last factor can have negative means. """
    for index in range(len(factors) - 1):
        meann = np.sign(np.mean(factors[index], axis=0))
        assert meann.size == factors[0].shape[1]

        factors[index] *= meann
        factors[index + 1] *= meann

    return factors


#### Decomposition Methods ###################################################################


def partial_tucker_decomp(tensor, mode_list, rank):
    """Perform Partial Tucker decomposition.
    -----------------------------------------------
    Input:
        tensor: 3D data tensor
        mode_list: which mode(s) to apply tucker decomposition to
        rank: rank of decomposition
    Returns
        output[0]: core tensor
        output[1]: list of factor matrices
    """
    return partial_tucker(tensor, mode_list, rank, tol=1.0e-12)

#### For R2X Plots ###########################################################################


def find_R2X_partialtucker(tucker_output, orig):
    """Compute R2X for the tucker decomposition."""
    return R2X(mode_dot(tucker_output[0], tucker_output[1][0], 2), orig)

###### To Flip Factors #########################################################################


def flip_factors(tucker_output):
    """For partial tucker OHSU factorization, flips protein and treatment/time factors if both negative for important values"""
    for component in range(tucker_output[0].shape[2]):
        av = 0.0
        for i in range(tucker_output[0].shape[0]):
            av += np.mean(tucker_output[0][i][:, component] ** 5)

        if av < 0 and tucker_output[1][0][:, component].mean() < 0:
            tucker_output[1][0][:, component] *= -1
            for j in range(tucker_output[0].shape[0]):
                tucker_output[0][j][:, component] *= -1
    return tucker_output

def R2X_OHSU(ax, p2slicesB):
    '''Creates R2X for OHSU PARAFAC2'''
    r2x = []
    compR2X = 10
    for i in range(1, compR2X + 1):
        parafac2tensor, error = OHSU_parafac2_decomp(p2slicesB, i)
        r2x.append(R2Xparafac2(p2slicesB, parafac2tensor))
    df = pd.DataFrame(r2x)
    comps = []
    for i in range(1, compR2X + 1):
        comps.append(i)
    df['Component'] = comps
    df.columns = ['Chromosomes', 'IFproteins', 'Histones', 'Gene Expression', 'RNA Genes', 'RPPA Proteins', 'Component']
    test = pd.melt(df, id_vars=['Component'])
    b = sns.scatterplot(data=test, x='Component', y='value', hue='variable', style='variable', ax=ax, s=100)
    b.set_xlabel("Component", fontsize=20)
    b.set_ylabel("R2X", fontsize=20)
    b.set_title("OHSU PARAFAC2")
    b.tick_params(labelsize=15)
    plt.legend(prop={'size': 15})
    ax.set_ylim(0, 1)

def OHSU_comp_plots(df, comps, ax):
    '''Plots treatments by each component from PARAFAC2'''
    sns.lineplot(data=df, x="Times", y=str(comps), hue="Treatments", ax=ax)
    ax.set_xlabel('Time (hr)')
    ax.set_title('Component ' + str(comps))

def R2Xparafac2(tensor_slices, decomposition):
    """Calculate the R2X of parafac2 decomposition"""
    R2Xp = np.zeros(len(tensor_slices))
    for idx, tensor_slice in enumerate(tensor_slices):
        reconstruction = parafac2_to_slice(decomposition, idx, validate=False)
        R2Xp[idx] = 1.0 - tl_var(reconstruction - tensor_slice) / tl_var(tensor_slice)
    return R2Xp
def OHSU_parafac2_decomp(tensorSlice, rank):
    """Perform PARAFAC2 decomposition.
    -----------------------------------------------
    Input:
        tensor: 3D data tensor
        rank: rank of decomposition
    Returns
        output[0]: PARAFAC2 tensor, decomp[0] = weights, decomp[1] = factors, decomp[2] = projection matricies
        output[1]: reconstruction error
    """
    decomp, error = parafac2(tensorSlice, rank, n_iter_max=100, return_errors=True, random_state=1)
    return decomp, error

#### For PARAFAC2 Projections to Factors ####################################################

def projections_to_factors(parafac2_decomp):
    '''Computes PARAFAC2 projections into factors'''
    weights, transform = apply_parafac2_projections(parafac2_decomp)
    return weights, transform

