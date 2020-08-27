#!/usr/bin/env python
#
# Copyright 2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Code snippet for SNP Zygosity Predictor- Training."""

# Import nemo and variantworks modules
import nemo
import os
import pathlib

from variantworks.dataloader import ReadPileupDataLoader
from variantworks.io.vcfio import VCFReader
from variantworks.networks import AlexNet
from variantworks.encoders import PileupEncoder, ZygosityLabelEncoder

# Get VariantWorks root directory
repo_root_dir = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

# Create neural factory
nf = nemo.core.NeuralModuleFactory(
    placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir="./")

# Create pileup encoder by selecting layers to encode. More encoding layers
# can be found in the documentation for PilupEncoder class.
encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY]
pileup_encoder = PileupEncoder(
    window_size=100, max_reads=100, layers=encoding_layers)

# Instantiate a zygosity encoder that generates output labels. Converts a variant entry
# into a class label for no variant, homozygous variant or heterozygous variant.
zyg_encoder = ZygosityLabelEncoder()

# Create neural network that receives 2 channel inputs (encoding layers defined above)
# and outputs a logit over three classes (no variant, homozygous variant, heterozygous variant.
model = AlexNet(num_input_channels=len(encoding_layers), num_output_logits=3)

# Get datasets to train on.
# NOTE: To train a neural network well, the model needs to see samples from all types of classes.
# The example here shows a file that has true variant (either homozygous or heterozygous),
# but in practice one also needs to pass a set of false positive samples so the model can learn to
# ignore them. False positive samples can be marked with `is_fp` so the reader can appripriately
# assign their variant types.
data_folder = os.path.join(repo_root_dir, "tests", "data")
bam = os.path.join(data_folder, "small_bam.bam")
samples = os.path.join(data_folder, "candidates.vcf.gz")
vcf_loader = VCFReader(vcf=samples, bams=[bam], is_fp=False)

# Create a data loader with custom sample and label encoder.
dataset_train = ReadPileupDataLoader(ReadPileupDataLoader.Type.TRAIN, [vcf_loader],
                                     batch_size=32, shuffle=True,
                                     sample_encoder=pileup_encoder, label_encoder=zyg_encoder)

# Use CrossEntropyLoss to train.
vz_ce_loss = nemo.backends.pytorch.common.losses.CrossEntropyLossNM(logits_ndim=2)

# Create NeMo training DAG.
vz_labels, encoding = dataset_train()
vz = model(encoding=encoding)
vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)

# Logger callback
logger_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[vz_loss],
    print_func=lambda x: nemo.logging.info(f'Train Loss: {str(x[0].item())}')
)

# Checkpointing models through NeMo callback
checkpoint_callback = nemo.core.CheckpointCallback(
    folder='./',
    load_from_folder=None,
    # Checkpointing frequency in steps
    step_freq=-1,
    # Checkpointing frequency in epochs
    epoch_freq=1,
    # Number of checkpoints to keep
    checkpoints_to_keep=1,
    # If True, CheckpointCallback will raise an Error if restoring fails
    force_load=False
)

# Kick off training
nf.train([vz_loss],
         callbacks=[logger_callback, checkpoint_callback],
         optimization_params={"num_epochs": 10, "lr": 0.001},
         optimizer="adam")
