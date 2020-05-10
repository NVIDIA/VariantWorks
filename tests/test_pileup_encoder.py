import os
import pytest
import torch

import nemo

from claragenomics.variantworks.variant_encoder import SnpPileupEncoder

from test_utils import get_data_folder

def test_snp_pileup_encoder():
    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU)

    max_reads = 100
    window_size = 5
    width = 2 * window_size + 1
    height = max_reads
    channels = {"reads"}

    pileup_encoder = SnpPileupEncoder(window_size = window_size, max_reads = max_reads, channels = channels)
    assert(pileup_encoder.size == (len(channels), height, width))

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    pileup = pileup_encoder.encode(bam, "1", 240000)
    assert(pileup.size() == torch.Size([len(channels), height, width]))
