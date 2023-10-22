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

class Generator:

    def __init__(self, simulator, sample_sizes, seed):
        self.simulator = simulator
        self.sample_sizes = sample_sizes
        self.rng = np.random.default_rng(seed)

        self.curr_params = None


    def simulate_batch(self, batch_size: int, root_dist: np.ndarray, mutator_threshold: float = 0.01, plot: bool = False):
        """
        """

        # initialize matrix in which to store data

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
        parameters = params.ParamSet()

        # simulate each region
        for i in tqdm.tqdm(range(batch_size)):
            # simulate tree sequence using simulation parameters.
            # NOTE: the simulator simulates numbers of *samples* rather
            # than haplotypes, so we need to divide the sample sizes by 2
            # to get the correct number of haplotypes.

            #root_dist_adj = util.add_noise(root_dist)
            ts = self.simulator(
                parameters,
                self.sample_sizes,
                root_dist,
                self.rng,
                mutator_threshold=mutator_threshold,
                plot=True if plot and not plotted else False,
            )
            # return 3D array
            region, positions = prep_simulated_region(ts)
            assert region.shape[0] == positions.shape[0]
            region_formatted = util.process_region(region, positions)
            regions[i] = region_formatted
            plotted = True

        return regions


def prep_simulated_region(ts) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    # the genotype matrix returned by tskit is our expected output
    # n_snps x n_haps matrix. however, even multi-allelic variants
    # are encoded 0/1, regardless of the allele.

    # the genotype matrix returned by tskit is our expected output
    # n_snps x n_haps matrix

    # strip singletons
    # Identify sites with a singleton allele
    sites_with_a_singleton_allele = []
    for v in ts.variants():
        non_missing_genotypes = v.genotypes[v.genotypes != tskit.MISSING_DATA]
        if np.any(np.bincount(non_missing_genotypes) == 1):
            sites_with_a_singleton_allele.append(v.site.id)
            
    # Strip those sites from the tree sequence
    ts = ts.delete_sites(sites_with_a_singleton_allele)

    n_snps, n_haps = ts.genotype_matrix().astype(np.float32).shape
    X = np.zeros((n_snps, n_haps, 6), dtype=np.float32)

    #X = ts.genotype_matrix().astype(np.float32)

    for vi, var in enumerate(ts.variants()):
        ref = var.alleles[0]
        alt_alleles = var.alleles[1:]
        gts = var.genotypes
        alt = alt_alleles[0]
        # ignore multi-allelics
        assert len(alt_alleles) == 1
        # for alt_i, alt in enumerate(alt_alleles):
        if ref in ("G", "T"):
            ref, alt = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
        # get indices of samples with this mutation
        mutation = ">".join([ref, alt])
        mutation_idx = global_vars.MUT2IDX[mutation]
        X[vi, :, mutation_idx] = gts

    site_table = ts.tables.sites
    positions = site_table.position.astype(np.int64)
    assert positions.shape[0] == X.shape[0]

    return X, positions

# testing
if __name__ == "__main__":

    batch_size = 10
    parameters = params.ParamSet()

    rng = np.random.default_rng(global_vars.DEFAULT_SEED)

    # quick test
    print("sim exp")
    generator = Generator(simulation.simulate_exp, 20, rng)

    mini_batch = generator.simulate_batch(batch_size)

    print("x", mini_batch.shape)