import pytest
import torch

from claragenomics.variantworks.pileup_encoder import SnpPileupGenerator

def test_snp_pileup_encoder():
    pileup_generator = SnpPileupGenerator(window_size = 10, max_reads = 100, channels = {"reads", "qscore"})
    assert(pileup_generator.size == (2, 100, 21))

    pileup = pileup_generator(None, None, None)
    assert(pileup.size() == torch.Size([2, 100, 21]))
