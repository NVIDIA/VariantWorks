# Classes and functions to encode pileups

import pysam
import torch

class SnpPileupGenerator:
    def __init__(self, window_size = 50, max_reads = 50, channels={"reads"}):
        self.window_size = window_size
        self.max_reads = max_reads
        self.channels = channels

    @property
    def size(self):
        return (len(self.channels), self.max_reads, 2 * self.window_size + 1)

    def __call__(self, bam, chrom, variant_pos):
        # Create an N-D pileup of reads covering a variant locus
        # TODO: Return proper pileup from SAM file
        pileup = torch.randint(1, 6, self.size, dtype=torch.float32)
        return pileup
