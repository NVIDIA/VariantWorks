import pytest
import os

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM

from claragenomics.variantworks.dataset import SnpPileupDataType
from claragenomics.variantworks.pileup_generator import SnpPileupGenerator
from claragenomics.variantworks.networks import AlexNet

def test_simple_vc():
    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU)

    # Generate dataset
    pileup_generator = SnpPileupGenerator(window_size = 200, max_reads = 1000, channels={"reads"})
    dataset = SnpPileupDataType(None, None, pileup_generator, batch_size = 32, shuffle = True)

    # Setup loss
    ce_loss = CrossEntropyLossNM(logits_dim=2)

    # Neural Network
    alexnet = AlexNet(num_channels=1, num_classes=3)

    # Create DAG
    pileups, labels = dataset()
    p = alexnet(pileup=pileups)
    loss = ce_loss(logits=p, labels=labels)

    # SimpleLossLoggerCallback will print loss values to console.
    callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss], print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}')
            )

    # Invoke the "train" action.
    nf.train([loss], callbacks=[callback], optimization_params={"num_epochs": 3, "lr": 0.0003}, optimizer="sgd")
