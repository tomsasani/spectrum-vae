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
        samples=[msprime.SampleSet(sample_sizes[0], population="A", ploidy=2)],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mu = params.mu.proposal(rng)

    mutation_model = msprime.F84(root_distribution=root_dist, kappa=2)

    # figure out if we are going to augment the mutation rate
    # in this simulation with a local mutator allele
    mutator_prob = params.mutator_prob.proposal(rng)
    if mutator_prob <= mutator_threshold:
        # define mutation model
        mutation_model = msprime.F84(
            root_distribution=root_dist,
            kappa=params.kappa.proposal(rng),
        )
    
    # otherwise, simulate constant mutation rate across the region for all time
    mts = msprime.sim_mutations(
        ts,
        rate=mu,
        model=mutation_model,
        random_seed=seed,
        discrete_genome=False,
    )

    return mts

def simulate_exp_one_channel(
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
        samples=[msprime.SampleSet(sample_sizes[0], population="A", ploidy=2)],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mu = params.mu.proposal(rng)

    mutation_model = msprime.BinaryMutationModel(state_independent=False)

    # figure out if we are going to augment the mutation rate
    # in this simulation with a local mutator allele
    mutator_prob = params.mutator_prob.proposal(rng)
    if mutator_prob <= mutator_threshold:
        mu_effect_size = params.mutator_effect_size.proposal(rng)

        # then, define the period of time in which the mutator should be active
        mu_emergence = params.mutator_emergence.proposal(rng)

        # simulate at a low rate before the emergence
        # across the whole region
        mts = msprime.sim_mutations(
            ts,
            rate=mu,
            model=mutation_model,
            start_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )
        # and a higher rate afterward
        # define RateMap so that mutator is only active in small region
        mu_start_pos = params.mutator_start.proposal(rng)
        mu_length = params.mutator_length.proposal(rng)
        # chunks corresponding to mutator activity regions
        position_list = [
            0,
            mu_start_pos,
            mu_start_pos + mu_length,
            global_vars.L,
        ]

        adjusted_mu = mu * mu_effect_size

        # normal rate map
        rate_map = msprime.RateMap(
            position=position_list,
            rate=[mu, adjusted_mu, mu],
        )
        mts = msprime.sim_mutations(
            mts,
            rate=rate_map,
            model=mutation_model,
            end_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )

    else:
        # otherwise, simulate constant mutation rate across the region for all time
        mts = msprime.sim_mutations(
            ts,
            rate=mu,
            model=mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    return mts


def simulate_gough(params, sample_sizes, root_dist, rng: np.random.default_rng, plot: bool = False, mutator_threshold: float = 0.05):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

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
    gough_ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(sample_sizes[0], population="gough", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.mouse_rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )
    mainland_ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(sample_sizes[1], population="mainland", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.mouse_rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mu = params.mouse_mu.proposal(rng)
    normal_mutation_model = parameterize_mutation_model(root_dist)

    # simulate mutations on the mainland population with a normal
    # mutational model
    mainland_mts = msprime.sim_mutations(
            mainland_ts,
            rate=mu,
            model=normal_mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

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
            mutator_strength=mu_effect_size,
        )

        # simulate at a low rate before the emergence
        # across the whole region
        gough_mts = msprime.sim_mutations(
            gough_ts,
            rate=mu,
            model=normal_mutation_model,
            start_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )

        # and a higher rate afterward
        # define RateMap so that mutator is only active in small region
        mu_start_pos = params.mutator_start.proposal(rng)
        mu_length = params.mutator_length.proposal(rng)
        # chunks corresponding to mutator activity regions
        position_list = [
            0,
            mu_start_pos,
            mu_start_pos + mu_length,
            global_vars.L,
        ]

        # if mutator affects a single mutation type, adjust
        # the overall mutation rate by the proportion of mutations
        # belonging to that type (C>A). i.e., a 5x increase in the C>A
        # mutation rate will only increase the overall mutation rate by 0.5x
        lambdas = np.array([0.25, 0.1, 0.1, 0.1, 0.1, 0.35])
        ind_mus = lambdas * mu
        # if the mutation rate of a single type is increased,
        # what's the total mutation rate afterward?
        ind_mus[1] *= mu_effect_size
        adjusted_mu = np.sum(ind_mus)

        # normal rate map
        rate_map = msprime.RateMap(
            position=position_list,
            rate=[mu, adjusted_mu, mu],
        )
        print (position_list, [mu, adjusted_mu, mu], ind_mus)
        gough_mts = msprime.sim_mutations(
            gough_mts,
            rate=rate_map,
            model=mutator_mutation_model,
            end_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )

    else:
        # otherwise, simulate constant mutation rate across the region for all time
        gough_mts = msprime.sim_mutations(
            gough_ts,
            rate=mu,
            model=normal_mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    combined_mts = gough_mts.union(mainland_mts, [tskit.NULL for _ in mainland_mts.nodes()])

    return combined_mts


def simulate_gough_one_channel(
    params,
    sample_sizes,
    root_dist,
    rng: np.random.default_rng,
    plot: bool = False,
    mutator_threshold: float = 0.05,
):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

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
    gough_ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(sample_sizes[0], population="gough", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.mouse_rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )
    mainland_ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(sample_sizes[1], population="mainland", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.mouse_rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mu = params.mouse_mu.proposal(rng)
    normal_mutation_model = parameterize_mutation_model(root_dist)
    normal_mutation_model = msprime.F84(kappa=2)

    # simulate mutations on the mainland population with a normal
    # mutational model
    mainland_mts = msprime.sim_mutations(
            mainland_ts,
            rate=mu,
            model=normal_mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    # figure out if we are going to augment the mutation rate
    # in this simulation with a local mutator allele
    mutator_prob = params.mutator_prob.proposal(rng)
    if mutator_prob <= mutator_threshold:
        mu_effect_size = params.mutator_effect_size.proposal(rng)

        # then, define the period of time in which the mutator should be active
        mu_emergence = params.mutator_emergence.proposal(rng)

        # simulate at a low rate before the emergence
        # across the whole region
        gough_mts = msprime.sim_mutations(
            gough_ts,
            rate=mu,
            model=normal_mutation_model,
            start_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )

        # and a higher rate afterward
        # define RateMap so that mutator is only active in small region
        mu_start_pos = params.mutator_start.proposal(rng)
        mu_length = params.mutator_length.proposal(rng)
        # chunks corresponding to mutator activity regions
        position_list = [
            0,
            mu_start_pos,
            mu_start_pos + mu_length,
            global_vars.L,
        ]

        adjusted_mu = mu * mu_effect_size

        # normal rate map
        rate_map = msprime.RateMap(
            position=position_list,
            rate=[mu, adjusted_mu, mu],
        )
        gough_mts = msprime.sim_mutations(
            gough_mts,
            rate=rate_map,
            model=normal_mutation_model,
            end_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )

    else:
        # otherwise, simulate constant mutation rate across the region for all time
        gough_mts = msprime.sim_mutations(
            gough_ts,
            rate=mu,
            model=normal_mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    combined_mts = gough_mts.union(mainland_mts, [tskit.NULL for _ in mainland_mts.nodes()])

    return combined_mts


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

    demography.add_population(name="A", initial_size=params.N1.proposal(rng), growth_rate=0,)
    demography.add_population(name="B", initial_size=params.N2.proposal(rng), growth_rate=0,)
    demography.add_population(name="ancestral", initial_size=params.N_anc.proposal(rng), growth_rate=0,)

    # directional (pulse)
    if params.mig.proposal(rng) >= 0:
        # migration from pop 1 into pop 0 (back in time)
        demography.add_mass_migration(
            time=params.T2.proposal(rng) / 2,
            source="A",
            dest="B",
            proportion=abs(params.mig.proposal(rng)),
        )
    else:
        # migration from pop 0 into pop 1 (back in time)
        demography.add_mass_migration(
            time=params.T2.proposal(rng) / 2,
            source="B",
            dest="A",
            proportion=abs(params.mig.proposal(rng)),
        )

    demography.add_population_split(time=params.T1.proposal(rng), derived=["A", "B"], ancestral="ancestral")

    if plot:
        graph = msprime.Demography.to_demes(demography)
        f, ax = plt.subplots()  # use plt.rcParams["figure.figsize"]
        demesdraw.tubes(graph, ax=ax, seed=1)
        f.savefig('im_demography.png', dpi=200)

    ts_a = msprime.sim_ancestry(
        #samples=sum(sample_sizes),
        samples=[
            msprime.SampleSet(sample_sizes[0], population="A", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    ts_b = msprime.sim_ancestry(
        #samples=sum(sample_sizes),
        samples=[
            msprime.SampleSet(sample_sizes[1], population="B", ploidy=2),
        ],
        demography=demography,
        sequence_length=global_vars.L,
        recombination_rate=params.rho.proposal(rng),
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mutation_model = msprime.BinaryMutationModel(state_independent=False)
    mu = params.mu.proposal(rng)

    mts_b = msprime.sim_mutations(
            ts_b,
            rate=mu,
            model=mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    mutator_prob = params.mutator_prob.proposal(rng)
    if mutator_prob <= mutator_threshold:
        mu_effect_size = params.mutator_effect_size.proposal(rng)

        # then, define the period of time in which the mutator should be active
        mu_emergence = params.mutator_emergence.proposal(rng)

        # simulate at a low rate before the emergence
        # across the whole region
        mts_a = msprime.sim_mutations(
            ts_a,
            rate=mu,
            model=mutation_model,
            start_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )

        # and a higher rate afterward
        # define RateMap so that mutator is only active in small region
        mu_start_pos = params.mutator_start.proposal(rng)
        mu_length = params.mutator_length.proposal(rng)
        # chunks corresponding to mutator activity regions
        position_list = [
            0,
            mu_start_pos,
            mu_start_pos + mu_length,
            global_vars.L,
        ]

        adjusted_mu = mu * mu_effect_size

        # normal rate map
        rate_map = msprime.RateMap(
            position=position_list,
            rate=[mu, adjusted_mu, mu],
        )
        mts_a = msprime.sim_mutations(
            mts_a,
            rate=rate_map,
            model=mutation_model,
            end_time=mu_emergence,
            random_seed=seed,
            discrete_genome=False,
        )

    else:
        # otherwise, simulate constant mutation rate across the region for all time
        mts_a = msprime.sim_mutations(
            ts_a,
            rate=mu,
            model=mutation_model,
            random_seed=seed,
            discrete_genome=False,
        )

    combined_mts = mts_a.union(mts_b, [tskit.NULL for _ in mts_b.nodes()])

    return combined_mts
