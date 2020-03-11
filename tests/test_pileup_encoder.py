import pytest
import torch

from claragenomics.variantworks.pileup_encoder import encode_snp_pileup

def test_snp_pileup_encoder():
    pileup = encode_snp_pileup(100, None, window_size = 10,
            max_reads = 100, channels = {"reads", "qscore"})
    assert(pileup.size() == torch.Size([100, 21, 2]))
