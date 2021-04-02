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
"""A sample program highlighting usage of VariantWorks SDK to write a simple SNP variant caller using a CNN."""

import argparse

import os
import nemo
import torch

from variantworks.dataloader import ReadPileupDataLoader
from variantworks.encoders import PileupEncoder, ZygosityLabelDecoder
from variantworks.io.vcfio import VCFReader, VCFWriter
from variantworks.networks import AlexNet


def create_model():
    """Return neural network to test."""
    # Neural Network
    alexnet = AlexNet(num_input_channels=2, num_output_logits=3)

    return alexnet


def infer(parsed_args):
    """Infer a sample model."""
    # Create neural factory as per NeMo requirements.
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU,
        checkpoint_dir=parsed_args.model_dir)

    vcf_readers = []
    for tp_file in parsed_args.tp_vcf_files:
        vcf_readers.append(VCFReader(vcf=tp_file, bams=[parsed_args.bam], is_fp=False))
    for fp_file in parsed_args.fp_vcf_files:
        vcf_readers.append(VCFReader(vcf=fp_file, bams=[parsed_args.bam], is_fp=True))

    # Setup encoder for samples and labels.
    sample_encoder = PileupEncoder(window_size=100, max_reads=100,
                                   layers=[PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY])
    test_dataset = ReadPileupDataLoader(ReadPileupDataLoader.Type.TEST, vcf_readers,
                                        batch_size=32, shuffle=False, sample_encoder=sample_encoder)

    model = create_model()

    encoding = test_dataset()

    # Execute inference
    vz = model(encoding=encoding)

    inferred_results = nf.infer([vz], checkpoint_dir=parsed_args.model_dir, verbose=True)

    # Decode inference results to labels
    inferred_zygosity = list()
    zyg_decoder = ZygosityLabelDecoder()
    for tensor_batches in inferred_results:
        for batch in tensor_batches:
            predicted_classes = torch.argmax(batch, dim=1)
            inferred_zygosity.extend([zyg_decoder(pred)
                                      for pred in predicted_classes])

    # Create output file for each vcf reader
    start_reader_idx = 0
    for vcf_reader in vcf_readers:
        input_vcf_df = vcf_reader.dataframe
        gt_col = "{}_GT".format(vcf_reader.samples[0])
        assert (gt_col in input_vcf_df)
        # Update GT column data
        reader_len = len(input_vcf_df[gt_col])
        input_vcf_df[gt_col] = inferred_zygosity[start_reader_idx:start_reader_idx+reader_len]
        start_reader_idx += reader_len
        output_path = '{}_{}.{}'.format(
            "inferred", "".join(os.path.basename(vcf_reader.file_path).split('.')[0:-1]), 'vcf')
        vcf_writer = VCFWriter(input_vcf_df, output_path=output_path, sample_names=vcf_reader.samples)
        vcf_writer.write_output(input_vcf_df)


def build_parser():
    """Build parser object with options for sample."""
    args_parser = argparse.ArgumentParser(
        description="Simple model inference SNP caller based on VariantWorks.")
    args_parser.add_argument("--tp-vcf-files", nargs="+",
                             help="List of TP VCF files to infer.", default=[], required=True)
    args_parser.add_argument("--fp-vcf-files", nargs="+",
                             help="List of FP VCF files to infer.", default=[])
    args_parser.add_argument("--bam", type=str,
                             help="BAM file with reads.", required=True)
    args_parser.add_argument("--model-dir", type=str,
                             help="Directory for loading saved trained model checkpoints.",
                             required=False, default="./models")
    return args_parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    infer(args)
