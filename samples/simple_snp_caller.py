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

import argparse
import os
import shutil
import tempfile
import time

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import compute_accuracy, eval_epochs_done_callback, eval_iter_callback
import torch

from claragenomics.variantworks.dataloader import ReadPileupDataLoader
from claragenomics.variantworks.io.vcfio import VCFReader
from claragenomics.variantworks.networks import AlexNet
from claragenomics.variantworks.sample_encoder import PileupEncoder, ZygosityLabelEncoder, ZygosityLabelDecoder


from distutils.dir_util import copy_tree


# Create temporary folder
tempdir = tempfile.mkdtemp()

def create_pileup_encoder_and_model():
    # Create encoder for variant
    encoding_layers = [PileupEncoder.Layer.READ]
    pileup_encoder = PileupEncoder(window_size=100, max_reads=100, layers=encoding_layers)
    
    # Neural Network
    alexnet = AlexNet(num_input_channels=len(encoding_layers), num_output_logits=3)

    return pileup_encoder, alexnet
    
def train():
    start_time = time.time()
    ######## Train
    # Train a sample model with test data
    
    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=tempdir)
   
    pileup_encoder, model = create_pileup_encoder_and_model()

    # Generate dataset
    bam = "/ssd/VariantWorks/end_to_end_workflow_sample_files/HG001.hs37d5.30x.bam"
    zyg_encoder = ZygosityLabelEncoder()
    
    # Setup loss
    
    # Create train DAG
    tp_vcf = VCFReader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/temp_tp.vcf.gz", bam=bam, is_fp=False)
    fp_vcf = VCFReader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/temp_fp.vcf.gz", bam=bam, is_fp=True)
    vcf_loader = VCFReader([tp_vcf, fp_vcf])
    print(len(vcf_loader))
    label_loader_time = time.time()
    print("Label loading time {}".format(label_loader_time - start_time))
    train_dataset = ReadPileupDataLoader(ReadPileupDataLoader.Type.TRAIN, vcf_loader, batch_size=32, shuffle=True, num_workers=32, sample_encoder=pileup_encoder, label_encoder=zyg_encoder)
    vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    vz_labels, encoding = train_dataset()
    vz = model(encoding=encoding)
    vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)
    
    # Create eval DAG
    eval_tp_vcf = VCFReader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/temp_tp.vcf.gz", bam=bam, is_fp=False)
    eval_fp_vcf = VCFReader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/temp_fp.vcf.gz", bam=bam, is_fp=True)
    eval_vcf_loader = VCFReader([eval_tp_vcf, eval_fp_vcf])
    print(len(eval_vcf_loader))
    label_loader_time = time.time()
    print("Label loading time {}".format(label_loader_time - start_time))
    eval_dataset = ReadPileupDataLoader(ReadPileupDataLoader.Type.EVAL, eval_vcf_loader, batch_size=128, shuffle=False, num_workers=32, sample_encoder=pileup_encoder, label_encoder=zyg_encoder)
    eval_vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    eval_vz_labels, eval_encoding = train_dataset()
    eval_vz = model(encoding=eval_encoding)
    eval_vz_loss = eval_vz_ce_loss(logits=eval_vz, labels=eval_vz_labels)
    
    # Logger callback
    loggercallback = nemo.core.SimpleLossLoggerCallback(
            tensors=[vz_loss, vz, vz_labels],
            step_freq=1,
            )
    
    # Checkpointing models through NeMo callback
    checkpointcallback = nemo.core.CheckpointCallback(
            folder=nf.checkpoint_dir,
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
    
    evaluator_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_vz_loss, eval_vz, eval_vz_labels],
            user_iter_callback=eval_iter_callback,
            user_epochs_done_callback=eval_epochs_done_callback,
            eval_step=1000,
            )
    
    # Invoke the "train" action.
    nf.train([vz_loss],
             callbacks=[loggercallback, checkpointcallback, evaluator_callback],
             optimization_params={"num_epochs": 4, "lr": 0.001},
             optimizer="adam")

def infer():
    ######## Inference
    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=tempdir)
    
    pileup_encoder, model = create_pileup_encoder_and_model()

    # Generate dataset
    bam = "/ssd/VariantWorks/end_to_end_workflow_sample_files/HG001.hs37d5.30x.bam"
    vcf_bam_tuple = VCFReader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/temp_fp.vcf.gz", bam=bam, is_fp=True)
    vcf_loader = VCFReader([vcf_bam_tuple])
    test_dataset = ReadPileupDataLoader(ReadPileupDataLoader.Type.TEST, vcf_loader, batch_size=32, shuffle=False, sample_encoder=pileup_encoder)
    
    # Create train DAG
    encoding = test_dataset()
    vz = model(encoding=encoding)
    
    # Invoke the "train" action.
    zyg_decoder = ZygosityLabelDecoder()
    results = nf.infer([vz], checkpoint_dir=tempdir, verbose=True)
    for tensor_batches in results:
        for batch in tensor_batches:
            predicted_classes = torch.argmax(batch, dim=1)
            for pred in predicted_classes:
                print(zyg_decoder(pred))
    
    shutil.rmtree(tempdir)

def main():
    train()
    infer()

if __name__ == "__main__":
    main()
