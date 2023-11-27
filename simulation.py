import msprime
import matplotlib.pyplot as plt
import demesdraw
import numpy as np

import math

import global_vars


def simulate_exp(
    params,
    sample_sizes,
    rng: np.random.default_rng,
    kappa: float = 2,
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
    mutation_model = msprime.F84(kappa=kappa)

    # otherwise, simulate constant mutation rate across the region for all time
    mts = msprime.sim_mutations(
        ts,
        rate=mu,
        model=mutation_model,
        random_seed=seed,
        discrete_genome=False,
    )

    return mts
