"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from sklearn.neighbors import NearestNeighbors

# our imports
import global_vars

def calculate_sfs(data: np.ndarray):
    n_batches, n_haps, n_snps, n_channels = data.shape
    sfs_arr = np.zeros((n_batches, n_haps // 2 - 1))
    acs = np.sum(data > 0, axis=1)
    df = []
    for batch_i in range(n_batches):
        sfs, _ = np.histogram(acs[batch_i], bins=np.arange(n_haps // 2))
        for ac_i, ac in enumerate(sfs):
            df.append({"batch": batch_i, "ac": ac_i, "count": ac})
    df = pd.DataFrame(df)
    return df.query("ac > 0")

def add_noise(root_dist: np.ndarray):
    # randomly add noise to the root distribution
    random_noise = np.random.normal(0., 0.01, size=4)
    root_dist_adj = root_dist + random_noise
    if np.any(root_dist_adj < 0):
        return add_noise(root_dist)
    else: return root_dist_adj

def inter_snp_distances(positions: np.ndarray, norm_len: int) -> np.ndarray:
    if positions.shape[0] > 0:
        dist_vec = [0]
        for i in range(positions.shape[0] - 1):
            # NOTE: inter-snp distances always normalized to simulated region size
            dist_vec.append((positions[i + 1] - positions[i]) / norm_len)
    else: dist_vec = []
    return np.array(dist_vec)


def sort_min_diff(X: np.ndarray):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''
    # reduce to 2 dims
    assert len(X.shape) == 3
    X_copy = np.copy(X)
    X_copy[X_copy < 0] = 0
    if X.shape[-1] == 6:
        X_copy = np.sum(X_copy, axis=2)
    else: X_copy = X_copy[:, :, 0]
    mb = NearestNeighbors(n_neighbors=len(X_copy), metric='manhattan').fit(X_copy)
    v = mb.kneighbors(X_copy)
    smallest = np.argmin(v[0].sum(axis=1))
    return v[1][smallest]


def find_segregating_idxs(X: np.ndarray):

    n_snps, n_haps = X.shape
    # initialize mask to store "good" sites
    to_keep = np.ones(n_snps)
    # find sites where there are bi-allelics
    multi_allelic = np.where(np.any(X > 1, axis=1))[0]
    if multi_allelic.shape[0] > 0: print ("found multi-allelic")
    to_keep[multi_allelic] = 0
    # remove sites that are non-segregating (i.e., if we didn't
    # add any information to them because they were multi-allelic
    # or because they were a silent mutation)
    acs = np.count_nonzero(X, axis=1)
    non_segregating = np.where((acs == 0) | (acs == n_haps))[0]
    if non_segregating.shape[0] > 0: print ("found non seg")
    to_keep[non_segregating] = 0
    return np.where(to_keep)[0]


def process_region(
    X: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """
    Process an array of shape (n_sites, n_haps, 6), which is produced
    from either generated or real data. First, subset it to contain global_vars.NUM_SNPS
    polymorphisms, and then calculate the sums of derived alleles on each haplotype in global_vars.N_WINDOWS
    windows across the arrays. 
    
    Zero-pad if necessary.

    Args:
        X (np.ndarray): feature array of shape (n_sites, n_haps, n_channels - 1)
        positions (np.ndarray): array of positions that correspond to each of the
            n_sites in the feature array.

    Returns:
        np.ndarray: _description_
    """
    # figure out how many sites and haplotypes are in the actual
    # multi-dimensional array
    n_sites, n_haps, n_channels = X.shape
    # make sure we have exactly as many positions as there are sites
    assert n_sites == positions.shape[0]

    # check for multi-allelics
    #assert np.max(X) == 1

    # figure out the half-way point (measured in numbers of sites)
    # in the input array
    mid = n_sites // 2
    half_S = global_vars.NUM_SNPS // 2

    # instantiate the new region, formatted as (n_haps, n_sites, n_channels)
    region = np.zeros(
        (n_haps, global_vars.NUM_SNPS, global_vars.NUM_CHANNELS),
        dtype=np.float32,
    )

    # should we divide by the *actual* region length?
    distances = inter_snp_distances(positions, global_vars.L)

    # first, transpose the full input matrix to be n_haps x n_snps
    X = np.transpose(X, (1, 0, 2))

    # sum across channels
    X = np.expand_dims(np.sum(X, axis=2), axis=2)

    # if we have more than the necessary number of SNPs
    if mid >= half_S:
        # define indices to use for slicing
        i, j = mid - half_S, mid + half_S
        # add sites to output
        region[:, :, :] = major_minor(X[:, i:j, :])
        # add one-hot to output
        # tile the inter-snp distances down the haplotypes
        # get inter-SNP distances, relative to the simualted region size
        distances_tiled = np.tile(distances[i:j], (n_haps, 1))
        # add final channel of inter-snp distances
        #region[:, :, -1] = distances_tiled

    else:
        other_half_S = half_S + 1 if n_sites % 2 == 1 else half_S
        i, j = half_S - mid, mid + other_half_S
        # use the complete genotype array
        # but just add it to the center of the main array
        region[:, i:j, :] = major_minor(X)
        # add one-hot to output
        # tile the inter-snp distances down the haplotypes
        distances_tiled = np.tile(distances, (n_haps, 1))
        # add final channel of inter-snp distances
        #region[:, i:j, -1] = distances_tiled


    return region


def major_minor(matrix):
    """Note that matrix.shape[1] may not be S if we don't have enough SNPs"""

    # NOTE: need to fix potential mispolarization if using ancestral genome?
    n_haps, n_sites, n_channels = matrix.shape
    
    # figure out the channel in which each mutation occurred
    for site_i in range(n_sites):
        #channel_haplotypes = matrix[:, site_i, :]        
        for mut_i in range(n_channels):
            # in this channel, figure out whether this site has any derived alleles
            haplotypes = matrix[:, site_i, mut_i]
            # if not, we'll mask all haplotypes at this site on this channel,
            # leaving the channel with the actual mutation unmasked
            if np.count_nonzero(haplotypes) == 0: continue
            else:
                # if there are derived alleles and there are more derived than ancestral,
                # flip the polarization
                if np.count_nonzero(haplotypes) > (n_haps / 2):
                    # if greater than 50% of haplotypes are ALT, reverse
                    # the REF/ALT polarization
                    haplotypes = 1 - haplotypes
                    haplotypes[haplotypes == 0] = -1
                    matrix[:, site_i, mut_i] = haplotypes
                # if there are fewer derived than ancestral, keep the haplotypes as is
                else:
                    haplotypes[haplotypes == 0] = -1
                    matrix[:, site_i, mut_i] = haplotypes
    return matrix