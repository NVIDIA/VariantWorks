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
"""Code snippet for SNP Zygosity Predictor- Inference."""

# Import nemo and variantworks modules
import nemo
import os
import pathlib
import torch

from variantworks.dataloader import VariantDataLoader
from variantworks.io.vcfio import VCFReader, VCFWriter
from variantworks.networks import AlexNet
from variantworks.encoders import PileupEncoder, ZygosityLabelDecoder

# Get VariantWorks root directory
repo_root_dir = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

# Create neural factory. In this case, the checkpoint_dir has to be set for NeMo to pick
# up a pre-trained model.
nf = nemo.core.NeuralModuleFactory(
    placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir="./")

# Dataset generation is done in a similar manner. It's important to note that the encoder used
# for inference much match that for training.
encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY]
pileup_encoder = PileupEncoder(
    window_size=100, max_reads=100, layers=encoding_layers)

# Neural Network
model = AlexNet(num_input_channels=len(encoding_layers), num_output_logits=3)

# Similar to training, a dataloader needs to be setup for the relevant datasets. In the case of
# inference, it doesn't matter if the files are tagged as false positive or not. Each example will be
# evaluated by the network. For simplicity the example is using the same dataset from training.
# Note: No label encoder is required in inference.
data_folder = os.path.join(repo_root_dir, "tests", "data")
bam = os.path.join(data_folder, "small_bam.bam")
labels = os.path.join(data_folder, "candidates.vcf.gz")
vcf_loader = VCFReader(vcf=labels, bams=[bam], is_fp=False)
test_dataset = VariantDataLoader(VariantDataLoader.Type.TEST, [vcf_loader], batch_size=32,
                                 shuffle=False, input_encoder=pileup_encoder)

# Create inference DAG
encoding = test_dataset()
vz = model(encoding=encoding)

# Invoke the "infer" action.
results = nf.infer([vz], checkpoint_dir="./", verbose=True)

# Instantiate a decoder that converts the predicted output of the network to
# a zygosity enum.
zyg_decoder = ZygosityLabelDecoder()

# Decode inference results to labels
inferred_zygosity = []
for tensor_batches in results:
    for batch in tensor_batches:
        predicted_classes = torch.argmax(batch, dim=1)
        inferred_zygosity += [zyg_decoder(pred)
                              for pred in predicted_classes]

# Update genotype entry in dataframe with predicted values.
genotype_column = "{}_GT".format(*vcf_loader.samples)
vcf_loader.dataframe[genotype_column] = inferred_zygosity

# Use the VCFWriter to output predicted zygosities to a VCF file.
result_writer = VCFWriter(vcf_loader.dataframe,
                          output_path="./out.vcf",
                          sample_names = vcf_loader.samples)

result_writer.write_output(vcf_loader.dataframe)
