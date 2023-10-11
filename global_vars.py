'''For collecting global values'''
# section A: general -----------------------------------------------------------
NUM_SNPS = 36 * 1      # number of seg sites, should be divisible by 4
L = 50_000
NUM_HAPLOTYPES = 50
NUM_CHANNELS = 1

DEFAULT_SEED = 1833
DEFAULT_SAMPLE_SIZE = 198

MUT2IDX = dict(zip(["C>T", "C>G", "C>A", "A>T", "A>C", "A>G"], range(6)))
REVCOMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
NUC_ORDER = ["A", "C", "G", "T"]
NUC2IDX = dict(zip(NUC_ORDER, range(4)))