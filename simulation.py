import msprime
import matplotlib.pyplot as plt
import demesdraw
import numpy as np
import tskit

import math

import global_vars


def get_transition_matrix(mutator_strength: float = 2.):

    # define expected mutation probabilities
    mutations = ["C>T", "C>A", "C>G", "A>T", "A>C", "A>G"]
    lambdas = np.array([25, 10, 10, 10, 10, 35])

    lambdas[1] *= mutator_strength
    lambdas = lambdas / np.sum(lambdas)

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

def parameterize_mutation_model(root_dist: np.ndarray, mutator_strength: float = 1.):

    norm_transition_matrix = get_transition_matrix(mutator_strength=mutator_strength)

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
    root_dist,
    rng: np.random.default_rng,
    plot: bool = False,
    mutator_threshold: float = 0.01,
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
        f.savefig('demography.png', dpi=200)

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
    mutation_model = msprime.F84(root_distribution=root_dist, kappa=params.kappa.value)

    # otherwise, simulate constant mutation rate across the region for all time
    mts = msprime.sim_mutations(
        ts,
        rate=mu,
        model=mutation_model,
        random_seed=seed,
        discrete_genome=False,
    )

    return mts


def simulate_im(
    params,
    sample_sizes,
    root_dist,
    rng,
    plot: bool = False,
    mutator_threshold: float = 0.05,
):

    seed = rng.integers(1, 2**32)

    demography = msprime.Demography()

    demography.add_population(name="A", initial_size=params.N1.value, growth_rate=0,)
    demography.add_population(name="B", initial_size=params.N2.value, growth_rate=0,)
    demography.add_population(
        name="ancestral",
        initial_size=params.N_anc.value,
        growth_rate=0,
    )

    # directional (pulse)
    if params.mig.proposal(rng) >= 0:
        # migration from pop 1 into pop 0 (back in time)
        demography.add_mass_migration(
            time=params.T2.value / 2,
            source="A",
            dest="B",
            proportion=abs(params.mig.value),
        )
    else:
        # migration from pop 0 into pop 1 (back in time)
        demography.add_mass_migration(
            time=params.T2.value / 2,
            source="B",
            dest="A",
            proportion=abs(params.mig.value),
        )

    demography.add_population_split(
        time=params.T1.value,
        derived=["A", "B"],
        ancestral="ancestral",
    )

    if plot:
        graph = msprime.Demography.to_demes(demography)
        f, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
        demesdraw.tubes(graph, ax=ax, seed=1)
        f.savefig('im_demography.png', dpi=200)

    ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(sample_sizes[0], population="A", ploidy=2),
            msprime.SampleSet(sample_sizes[1], population="B", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.value,
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mutation_model = msprime.F84(root_distribution=root_dist, kappa=2)
    mu = params.mu.value

    mts = msprime.sim_mutations(
            ts,
            rate=mu,
            model=mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )


    return mts
