import os
import pytest
import torch

from claragenomics.variantworks.pileup_generator import SnpPileupGenerator

from test_utils import get_data_folder

def test_snp_pileup_generator():
    max_reads = 100
    window_size = 5
    width = 2 * window_size + 1
    height = max_reads
    channels = {"reads"}

    pileup_generator = SnpPileupGenerator(window_size = window_size, max_reads = max_reads, channels = channels)
    assert(pileup_generator.size == (len(channels), height, width))

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    pileup = pileup_generator(bam, "1", 240000)
    assert(pileup.size() == torch.Size([len(channels), height, width]))
