'''For collecting global values'''
# section A: general -----------------------------------------------------------
NUM_SNPS = 64   # number of seg sites, should be divisible by 4
L = 50_000
NUM_HAPLOTYPES = 198
NUM_CHANNELS = 6

DEFAULT_SEED = 1833

MUT2IDX = dict(zip(["C>T", "C>G", "C>A", "A>T", "A>C", "A>G"], range(6)))
REVCOMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
NUC_ORDER = ["A", "C", "G", "T"]
NUC2IDX = dict(zip(NUC_ORDER, range(4)))
