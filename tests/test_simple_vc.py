import pytest
import os

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM

from claragenomics.variantworks.dataset import SnpPileupDataType
from claragenomics.variantworks.pileup_generator import SnpPileupGenerator
from claragenomics.variantworks.networks import AlexNet

from test_utils import get_data_folder

def test_simple_vc():
    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU)

    # Generate dataset
    pileup_generator = SnpPileupGenerator(window_size = 200, max_reads = 1000, channels={"reads"})

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    dataset = SnpPileupDataType(bam, None, pileup_generator, batch_size = 2, shuffle = True)

    # Setup loss
    vt_ce_loss = CrossEntropyLossNM(logits_dim=2)
    va_ce_loss = CrossEntropyLossNM(logits_dim=2)

    # Neural Network
    alexnet = AlexNet(num_input_channels=1, num_vt=3, num_alleles=4)

    # Create DAG
    pileups, vt_labels, va_labels = dataset()
    vt, va = alexnet(pileup=pileups)
    vt_loss = vt_ce_loss(logits=vt, labels=vt_labels)
    va_loss = va_ce_loss(logits=va, labels=va_labels)

    # SimpleLossLoggerCallback will print loss values to console.
    callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[vt_loss, va_loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}')
            )

    # Invoke the "train" action.
    nf.train([vt_loss, va_loss], callbacks=[callback], optimization_params={"num_epochs": 1, "lr": 0.0003}, optimizer="sgd")
