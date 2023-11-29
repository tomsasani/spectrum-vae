"""
Generator class for pg-gan.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# our imports
import global_vars
import params
import simulation
import util
import tqdm
import tskit
from typing import List, Union


class Generator:
    def __init__(self, simulator, sample_sizes, param_names, seed):
        self.simulator = simulator
        self.sample_sizes = sample_sizes
        self.param_names = param_names
        self.rng = np.random.default_rng(seed)

        self.curr_params = None

    def simulate_batch(
        self,
        batch_size: int,
        root_dist: np.ndarray,
        param_values: List[Union[float, int]] = [],
        treat_as_real: bool = False,
        effect_size: float = 1.,
        plot: bool = False,
        one_hot: bool = True,
    ):
        """ """

        # initialize matrix in which to store data
        if one_hot:
            regions = np.zeros(
                (
                    batch_size,
                    4,
                    global_vars.NUM_SNPS,
                    sum(self.sample_sizes) * 2,
                    2 + 1,
                ),
                dtype=np.float32,
            )
        else:
            regions = np.zeros(
                (
                    batch_size,
                    sum(self.sample_sizes) * 2,
                    global_vars.NUM_SNPS,
                    global_vars.NUM_CHANNELS,
                ),
                dtype=np.float32,
            )
        plotted = False

        # initialize the collection of parameters
        sim_params = params.ParamSet()
        if treat_as_real:
            pass
        elif params == []:
            sim_params.update(self.param_names, self.curr_params)
        else:
            sim_params.update(self.param_names, param_values)

        # simulate each region
        for i in tqdm.tqdm(range(batch_size)):
            # simulate tree sequence using simulation parameters.
            # NOTE: the simulator simulates numbers of *samples* rather
            # than haplotypes, so we need to divide the sample sizes by 2
            # to get the correct number of haplotypes.

            ts = self.simulator(
                sim_params,
                self.sample_sizes,
                root_dist,
                self.rng,
                effect_size=effect_size,
                plot=True if plot and not plotted else False,
            )
            # return 3D array
            region, positions = prep_simulated_region(ts, one_hot=one_hot)
            #assert region.shape[1] == positions.shape[0]
            if one_hot:
                region_formatted = util.process_region_onehot(region, positions)
            else:
                region_formatted = util.process_region(region, positions)
            regions[i] = region_formatted
            plotted = True

        return regions


def prep_simulated_region(ts, one_hot: bool = False) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    n_snps, n_haps = ts.genotype_matrix().astype(np.float32).shape
    # n_nucs, n_snps, n_haps, n_channels
    if one_hot:
        X = np.zeros((4, n_snps, n_haps, 2), dtype=np.float32)
    else:
        X = np.zeros((n_snps, n_haps, 6), dtype=np.float32)

    for vi, var in enumerate(ts.variants()):
        ref = var.alleles[0]
        alt_alleles = var.alleles[1:]
        gts = var.genotypes
        
        # ignore multi-allelics
        assert len(alt_alleles) == 1

        if one_hot:
            ref_idx = global_vars.NUC2IDX[var.alleles[0]]
            alt_idx = global_vars.NUC2IDX[alt_alleles[0]]

            has_alt = np.where(gts == 1)[0]
            has_ref = np.where(gts == 0)[0]
            # for every haplotype, increment the "REF" array
            # to have the reference nucleotide
            X[ref_idx, vi, :, 0] = 1
            # for every haplotype that has the alternate allele,
            # increment the "ALT" array to have the alt nucleotide
            X[alt_idx, vi, has_alt, 1] = 1
            # for every haplotype that has the ref allele,
            # increment the "ALT" array to have the ref nucleotide
            X[ref_idx, vi, has_ref, 1] = 1
        else:
            alt = alt_alleles[0]
            if ref in ("G", "T"):
                ref, alt = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
            # shouldn't be any silent mutations given transition matrix
            assert ref != alt
            mutation = ">".join([ref, alt])
            mutation_idx = global_vars.MUT2IDX[mutation]

            X[vi, :, mutation_idx] = gts

    site_table = ts.tables.sites
    positions = site_table.position.astype(np.int64)
    if one_hot: 
        assert positions.shape[0] == X.shape[1]
    else:
        assert positions.shape[0] == X.shape[0]

    return X, positions


# testing
if __name__ == "__main__":
    batch_size = 10
    parameters = params.ParamSet()

    rng = np.random.default_rng(global_vars.DEFAULT_SEED)

    # quick test
    print("sim exp")
    generator = Generator(simulation.simulate_exp, [20], [], 42)

    mini_batch = generator.simulate_batch(batch_size)

    f, (ax1, ax2, ax3) = plt.subplots(3)
    sns.heatmap(mini_batch[0, :, :, 0, 0], ax=ax1)
    sns.heatmap(mini_batch[0, :, :, 0, 1], ax=ax2)
    sns.heatmap(mini_batch[0, :, :, 0, 2], ax=ax3)

    f.savefig("one-hot.png")
    print("x", mini_batch.shape)
