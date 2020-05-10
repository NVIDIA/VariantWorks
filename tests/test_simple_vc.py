import pytest
import os

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import compute_accuracy

from claragenomics.variantworks.dataset import VariantDataLoader
from claragenomics.variantworks.variant_encoder import SnpPileupEncoder
from claragenomics.variantworks.networks import AlexNet

from test_utils import get_data_folder

def test_simple_vc():
    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU)

    # Generate dataset
    pileup_generator = SnpPileupEncoder(window_size = 100, max_reads = 100, channels={"reads"})

    bam = os.path.join(get_data_folder(), "small_bam.bam")
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    train_dataset = VariantDataLoader(bam, labels, batch_size = 32, shuffle = True)
    pileup_encoder = SnpPileupEncoder(window_size = 100, max_reads = 100, channels={"reads"})

    # Setup loss
    vt_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    va_ce_loss = CrossEntropyLossNM(logits_ndim=2)

    # Neural Network
    alexnet = AlexNet(num_input_channels=1, num_vt=3, num_alleles=5)

    # Create train DAG
    vt_labels, va_labels, variant_positions = train_dataset()
    pileups = pileup_encoder(variant_pos = variant_positions)
    vt, va = alexnet(pileup=pileups)
    vt_loss = vt_ce_loss(logits=vt, labels=vt_labels)
    va_loss = va_ce_loss(logits=va, labels=va_labels)


    # SimpleLossLoggerCallback will print loss values to console.
    def my_print_fn(x):
        va_output = x[2]
        va_labels = x[3]
        acc = compute_accuracy((x[1], va_output, va_labels))
        logging.info(f'Train VT Loss: {str(x[0].item())}, Train VA Loss: {str(x[1].item())}, Accuracy : {str(acc)}')

    callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[vt_loss, va_loss, va, va_labels],
            print_func=my_print_fn,
            step_freq=1,
            )

    # Invoke the "train" action.
    nf.train([vt_loss, va_loss], callbacks=[callback], optimization_params={"num_epochs": 10, "lr": 0.001}, optimizer="adam")
