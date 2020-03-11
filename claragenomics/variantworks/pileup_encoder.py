# Classes and functions to encode pileups

import pysam
import torch

def encode_snp_pileup(variant_pos, bamf_file, window_size = 50, max_reads = 50, channels={"reads"}):
    # Create an N-D pileup of reads covering a variant locus

    # TODO: Return proper pileup from SAM file
    pileup = torch.randint(1, 6, [max_reads, 2 * window_size + 1, len(channels)], dtype=torch.uint8)
    return pileup
