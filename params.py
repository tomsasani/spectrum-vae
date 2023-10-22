"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from scipy.stats import norm
import sys

import global_vars

class Parameter:
    """
    Holds information about evolutionary parameters to infer.
    Note: the value arg is NOT the starting value, just used as a default if
    that parameter is not inferred, or the truth when training data is simulated
    """

    def __init__(self, minimum, maximum, name):
        self.min = minimum
        self.max = maximum
        self.name = name


    def proposal(self, rng: np.random.default_rng):
        return rng.uniform(self.min, self.max)


class ParamSet:

    def __init__(self):

        # population sizes and bottleneck times
        self.N1 = Parameter(22_000, 22_000, "N1")
        self.N2 = Parameter(3300, 3300, "N2")
        self.T1 = Parameter(3500, 3500, "T1")
        self.T2 = Parameter(1000, 1000, "T2")
        self.N_anc = Parameter(15000, 15000, "N_anc")
        self.T_split = Parameter(2000, 2000, "T_split")
        self.mig = Parameter(0.05, 0.05, "mig")
        # recombination rate
        self.rho = Parameter(1.25e-8, 1.25e-8, "rho")
        # mutation rate
        self.mu = Parameter(1.25e-8, 1.25e-8, "mu")
        # population growth parameter
        self.growth = Parameter(5e-3, 5e-3, "growth")
        self.kappa = Parameter(2.1, 3, "kappa")
        self.mutator_prob = Parameter(0, 1, "mu_prob")
        self.mutator_effect_size = Parameter(1.5, 2, "mu_effect")
        self.mutator_duration = Parameter(100, 300, "mu_duration")
        self.mutator_emergence = Parameter(1, 3_500, "mu_emergence")
        self.mutator_start = Parameter(15_000, 15_000, "mu_start")
        self.mutator_length = Parameter(20_000, 20_000, "mu_length")

        self.N_gough = Parameter(20_000, 20_000, "N_gough")
        self.N_mainland = Parameter(50_000, 50_000, "N_mainland")
        self.T_colonization = Parameter(100, 100, "T_colonization")
        self.N_colonization = Parameter(1_000, 1_000, "N_colonization")
        self.T_mainland_bottleneck = Parameter(10_000, 10_000, "T_mainland_bottleneck")
        self.D_mainland_bottleneck = Parameter(6e-4, 6e-4, "D_mainland_bottleneck")
        self.island_migration_rate = Parameter(8e-4, 8e-4, "island_migration_rate")
        self.mouse_mu = Parameter(6.5e-9, 6.5e-9, "mouse_mu")
        self.mouse_rho = Parameter(1e-8, 1e-8, "mouse_rho")