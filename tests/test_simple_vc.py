import pytest
import os

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import compute_accuracy

from claragenomics.variantworks.dataset import VariantDataLoader
from claragenomics.variantworks.label_loader import VCFLabelLoader
from claragenomics.variantworks.networks import AlexNet
from claragenomics.variantworks.types import VcfBamPaths
from claragenomics.variantworks.variant_encoder import PileupEncoder, ZygosityLabelEncoder


from test_utils import get_data_folder


def test_simple_vc():
    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU)

    # Generate dataset
    encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY, PileupEncoder.Layer.MAPPING_QUALITY]
    pileup_encoder = PileupEncoder(window_size = 100, max_reads = 100, layers = encoding_layers)
    bam = os.path.join(get_data_folder(), "small_bam.bam")
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_bam_tuple = VcfBamPaths(vcf=labels, bam=bam, is_fp=False)
    vcf_loader = VCFLabelLoader([vcf_bam_tuple], allow_snps=True, allow_multiallele=False)
    zyg_encoder = ZygosityLabelEncoder()
    train_dataset = VariantDataLoader(pileup_encoder, vcf_loader, zyg_encoder, batch_size = 32, shuffle = True)

    # Setup loss
    vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)

    # Neural Network
    alexnet = AlexNet(num_input_channels=len(encoding_layers), num_vz=3)

    # Create train DAG
    vz_labels, encoding = train_dataset()
    vz = alexnet(encoding=encoding)
    vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)

    # SimpleLossLoggerCallback will print loss values to console.
    def my_print_fn(x):
        acc = compute_accuracy(x)
        logging.info(f'Train VT Loss: {str(x[0].item())}, Accuracy : {str(acc)}')

    callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[vz_loss, vz, vz_labels],
            print_func=my_print_fn,
            step_freq=1,
            )

    # Invoke the "train" action.
    nf.train([vz_loss], callbacks=[callback], optimization_params={"num_epochs": 10, "lr": 0.001}, optimizer="adam")
