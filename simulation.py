import msprime
import matplotlib.pyplot as plt
import demesdraw
import numpy as np

import math

import global_vars

def get_transition_matrix(effect_size: float = 1.):

    # define expected mutation probabilities
    mutations = ["C>T", "C>A", "C>G", "A>T", "A>C", "A>G"]
    lambdas = np.array([46, 17, 12, 1, 7.5, 17])
    lambdas[0] *= effect_size
    transition_matrix = np.zeros((4, 4))

    # for every mutation type...
    for mutation, prob in zip(mutations, lambdas):
        # get the indices of the reference and alt alleles
        ref, alt = mutation.split(">")
        # as well as the reverse complement
        ref_rc, alt_rc = global_vars.REVCOMP[ref], global_vars.REVCOMP[alt]
        # add its mutation probability to the transition matrix
        for r, a in ((ref, alt), (ref_rc, alt_rc)):
            ri, ai = global_vars.NUC2IDX[r], global_vars.NUC2IDX[a]
            transition_matrix[ri, ai] = prob

    # normalize transition matrix so that rows sum to 1
    rowsums = np.sum(transition_matrix, axis=1)
    norm_transition_matrix = transition_matrix / rowsums[:, np.newaxis]
    np.fill_diagonal(norm_transition_matrix, val=0)

    return norm_transition_matrix


def parameterize_mutation_model(root_dist: np.ndarray, effect_size: float = 1.):

    norm_transition_matrix = get_transition_matrix(effect_size=effect_size)

    if np.sum(root_dist) != 1:
        root_dist = root_dist / np.sum(root_dist)

    model = msprime.MatrixMutationModel(
        global_vars.NUC_ORDER,
        root_distribution=root_dist,
        transition_matrix=norm_transition_matrix,
    )

    return model


def simulate_exp(
    params,
    sample_sizes,
    root_dist: np.ndarray,
    rng: np.random.default_rng,
    effect_size: float = 1.,
    plot: bool = False,
):
    # get random seed for msprime simulations
    seed = rng.integers(1, 2**32)

    # get demographic parameters
    N1, N2 = params.N1.value, params.N2.value
    T1, T2 = params.T1.value, params.T2.value
    growth = params.growth.value

    N0 = N2 / math.exp(-growth * T2)

    demography = msprime.Demography()
    # at present moment, create population A with the size it should be
    # following its period of exponential growth
    demography.add_population(
        name="A",
        initial_size=N0,
        growth_rate=growth,
    )
    # T2 generations in the past, change the population size to be N2
    demography.add_population_parameters_change(
        population="A",
        time=T2,
        initial_size=N2,
        growth_rate=0,
    )

    # T1 generations in the past, change the population size to be N1
    demography.add_population_parameters_change(
        population="A",
        time=T1,
        initial_size=N1,
        growth_rate=0,
    )

    if plot:
        graph = msprime.Demography.to_demes(demography)
        f, ax = plt.subplots()
        demesdraw.tubes(graph, ax=ax, seed=1)
        f.savefig("demography.png", dpi=200)

    # sample sample_sizes diploids from the diploid population
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(sample_sizes[0], population="A", ploidy=2)],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.value,
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mu = params.mu.value
    # mutation_model = msprime.F84(kappa=kappa)
    mutation_model = parameterize_mutation_model(root_dist, effect_size=effect_size)
    # otherwise, simulate constant mutation rate across the region for all time
    mts = msprime.sim_mutations(
        ts,
        rate=mu,
        model=mutation_model,
        random_seed=seed,
        discrete_genome=False,
    )

    return mts
