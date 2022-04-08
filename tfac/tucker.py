""" This file includes functions to perform tucker decomposition. """
from tensorly.decomposition import tucker
import numpy as np
import pandas as pd
import itertools as it
import tensorly as tl


def tucker_decomp(tensor, num_comps):
    """ Performs Tucker decomposition. """

    ranks = tucker_rank(tensor.ndim, num_comps+1)

    total_r2x = []
    rank = []
    for total_rank in ranks:
        r2x = []
        for eachCP_rank in total_rank:
            fac = tucker(tensor.to_numpy(), rank=eachCP_rank, svd='randomized_svd')
            r2x.append(1 - ((tl.norm(tl.tucker_to_tensor(fac) - tensor.to_numpy()) ** 2) / tl.norm(tensor.to_numpy()) ** 2))
        total_r2x.append(max(r2x))
        rank.append(str(total_rank[r2x.index(max(r2x))]))

    return pd.DataFrame({"R2X": total_r2x}, index=rank)


def tucker_rank(first: int, last: int):
    """ Create list of tuples to pass to tucker as the rank. Since we have three dimensions, we need pairs of 3 numbers.
    :param first: least number of total components = the number of dimensions. here 3 (1, 1, 1)
    :param last: most number of total components = highest_rank - 1.
    Example:
    first = 3, last = 5 -> [[(1, 1, 1)], [(1, 1, 2), (1, 2, 1), (2, 1, 1)]]
    :return ranks: list of lists of permutations of ranks for each component.
    """

    ranks = []
    for i in range(first, last):
        choices = [pair for pair in it.combinations_with_replacement(range(1, last-1), 3) if sum(pair) == i]
        permutation = []
        for sets in choices:
            permutation.append(list(set([pars for pars in it.permutations(sets)])))
        ranks.append(list(it.chain(*permutation)))

    return ranks