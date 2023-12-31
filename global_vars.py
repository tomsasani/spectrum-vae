'''For collecting global values'''
# section A: general -----------------------------------------------------------
NUM_SNPS = 32   # number of seg sites, should be divisible by 4
L = 100_000
NUM_HAPLOTYPES = NUM_SNPS
NUM_CHANNELS = 3

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 3

DEFAULT_SEED = 1833

MUT2IDX = dict(zip(["C>T", "C>G", "C>A", "A>T", "A>C", "A>G"], range(6)))
REVCOMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
NUC_ORDER = ["A", "C", "G", "T"]
NUC2IDX = dict(zip(NUC_ORDER, range(4)))
