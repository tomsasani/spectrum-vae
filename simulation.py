import msprime
import matplotlib.pyplot as plt
import demesdraw
import numpy as np

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
    N1, N2 = params.N1.proposal(rng), params.N2.proposal(rng)
    T1, T2 = params.T1.proposal(rng), params.T2.proposal(rng)
    growth = params.growth.proposal(rng)

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
        samples=[msprime.SampleSet(sample_sizes[0], population="A", ploidy=1)],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mu = params.mu.proposal(rng)

    normal_mutation_model = parameterize_mutation_model(root_dist)

    # figure out if we are going to augment the mutation rate
    # in this simulation with a local mutator allele
    mutator_prob = params.mutator_prob.proposal(rng)
    if mutator_prob <= mutator_threshold:
        mu_effect_size = params.mutator_effect_size.proposal(rng)
        
        # then, define the period of time in which the mutator should be active
        mu_emergence = params.mutator_emergence.proposal(rng)

        # define mutation model
        mutator_mutation_model = parameterize_mutation_model(
            root_dist,
            mutator_strength=mu * mu_effect_size,
        )

        # simulate at a low rate before the emergence
        # across the whole region
        mts = msprime.sim_mutations(
            ts,
            rate=mu,
            model=mutator_mutation_model,
            #start_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )
        # and a higher rate afterward
        # mts = msprime.sim_mutations(
        #     mts,
        #     rate=mu,
        #     model=mutator_mutation_model,
        #     end_time=mu_emergence,
        #     random_seed=seed,
        #     discrete_genome=False,
        # )

    else:
        # otherwise, simulate constant mutation rate across the region for all time
        mts = msprime.sim_mutations(
            ts,
            rate=mu,
            model=normal_mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    return mts


def simulate_gough(params, sample_sizes, root_dist, rng: np.random.default_rng, plot: bool = False, mutator_threshold: float = 0.05):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # N_gough = params.N_bottleneck.value / math.exp(-params.growth.value * params.T_bottleneck.value)

    # size = n_bot / e^(-g * t_bot)

    # size * (e^(-g * t_bot)) = n_bot

    # e ^ (-g * t_bot) = n_bot / size

    # -g * t_bot = ln(n_bot / size)

    # -g = ln(n_bot / size) / t_bot

    # g = -1 * ln(n_bot / size) / t_bot
    seed = rng.integers(1, 2**32)

    N_gough, N_mainland = params.N_gough.proposal(rng), params.N_mainland.proposal(rng)
    N_colonization, T_colonization = params.N_colonization.proposal(rng), params.T_colonization.proposal(rng)
    island_migration_rate = params.island_migration_rate.proposal(rng)

    # calculate growth rate given colonization Ne and current Ne
    gough_growth = -1 * np.log(N_colonization / N_gough) / T_colonization

    demography = msprime.Demography()
    # at present moment, gough population which has grown exponentially
    demography.add_population(
        name="gough",
        initial_size=N_gough,
        growth_rate=gough_growth,
    )
    demography.add_population(
        name="mainland",
        initial_size=N_mainland,
        growth_rate=0,
    )
    demography.add_population(
        name="ancestral",
        initial_size=N_mainland,
        growth_rate=0,
    )
    demography.set_migration_rate(
        source="gough",
        dest="mainland",
        rate=island_migration_rate,
    )
    demography.add_population_split(
        time=T_colonization,
        derived=["gough", "mainland"],
        ancestral="ancestral",
    )

    # demography.add_simple_bottleneck(
    #     time=params.T_mainland_bottleneck.value,
    #     population="ancestral",
    #     proportion=params.D_mainland_bottleneck.value
    # )

    if plot:
        graph = msprime.Demography.to_demes(demography)
        f, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
        demesdraw.tubes(graph, ax=ax, seed=1)
        f.savefig('gough_demography.png', dpi=200)

    # sample sample_sizes monoploid haplotypes from the diploid population
    ts = msprime.sim_ancestry(
        #samples=sum(sample_sizes),
        samples=[
            msprime.SampleSet(sample_sizes[0], population="gough", ploidy=2),
            msprime.SampleSet(sample_sizes[1], population="mainland", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.mouse_rho.proposal(rng),
        discrete_genome=False,  # ensure no multi-allelics
        random_seed=seed,
        ploidy=2,
    )

    mu = params.mouse_mu.proposal(rng)

    normal_mutation_model = parameterize_mutation_model(root_dist)

    # figure out if we are going to augment the mutation rate
    # in this simulation with a local mutator allele
    mutator_prob = params.mutator_prob.proposal(rng)
    if mutator_prob <= mutator_threshold:
        mu_effect_size = params.mutator_effect_size.proposal(rng)
        
        # then, define the period of time in which the mutator should be active
        mu_emergence = params.mutator_emergence.proposal(rng)

        # define mutation model
        mutator_mutation_model = parameterize_mutation_model(
            root_dist,
            mutator_strength=mu * mu_effect_size,
        )

        # simulate at a low rate before the emergence
        # across the whole region
        mts = msprime.sim_mutations(
            ts,
            rate=mu,
            model=mutator_mutation_model,
            #start_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )
        # and a higher rate afterward
        # mts = msprime.sim_mutations(
        #     mts,
        #     rate=mu,
        #     model=mutator_mutation_model,
        #     end_time=mu_emergence,
        #     random_seed=seed,
        #     discrete_genome=False,
        # )

    else:
        # otherwise, simulate constant mutation rate across the region for all time
        mts = msprime.sim_mutations(
            ts,
            rate=mu,
            model=normal_mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    return mts