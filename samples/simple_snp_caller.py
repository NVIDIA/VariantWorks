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

import os
import shutil
import tempfile
import time

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import compute_accuracy, eval_epochs_done_callback, eval_iter_callback
import torch

from claragenomics.variantworks.dataset import VariantDataLoader
from claragenomics.variantworks.label_loader import VCFLabelLoader
from claragenomics.variantworks.networks import AlexNet
from claragenomics.variantworks.variant_encoder import PileupEncoder, ZygosityLabelEncoder


from distutils.dir_util import copy_tree

start_time = time.time()
######## Train
# Train a sample model with test data

# Create temporary folder
tempdir = tempfile.mkdtemp()

# Create neural factory
nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=tempdir)

# Generate dataset
encoding_layers = [PileupEncoder.Layer.READ]#, PileupEncoder.Layer.BASE_QUALITY, PileupEncoder.Layer.MAPPING_QUALITY]
pileup_encoder = PileupEncoder(window_size=100, max_reads=100, layers=encoding_layers)
bam = "/ssd/VariantWorks/end_to_end_workflow_sample_files/HG001.hs37d5.30x.bam"
zyg_encoder = ZygosityLabelEncoder()

# Setup loss

# Neural Network
alexnet = AlexNet(num_input_channels=len(encoding_layers), num_vz=3)

# Create train DAG
tp_vcf = VCFLabelLoader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/tp_vcf_1m_giab.vcf.gz", bam=bam, is_fp=False)
fp_vcf = VCFLabelLoader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/train_fp.vcf.gz", bam=bam, is_fp=True)
vcf_loader = VCFLabelLoader([tp_vcf, fp_vcf])
print(len(vcf_loader))
label_loader_time = time.time()
print("Label loading time {}".format(label_loader_time - start_time))
train_dataset = VariantDataLoader(pileup_encoder, vcf_loader, zyg_encoder, batch_size=32, shuffle=True, num_workers=32)
vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
vz_labels, encoding = train_dataset()
vz = alexnet(encoding=encoding)
vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)

# Create eval DAG
eval_tp_vcf = VCFLabelLoader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/test_tp_vcf_100k_giab.vcf.gz", bam=bam, is_fp=False)
eval_fp_vcf = VCFLabelLoader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/test_fp_vcf_100k_giab.vcf.gz", bam=bam, is_fp=True)
eval_vcf_loader = VCFLabelLoader([eval_tp_vcf, eval_fp_vcf])
print(len(eval_vcf_loader))
label_loader_time = time.time()
print("Label loading time {}".format(label_loader_time - start_time))
eval_dataset = VariantDataLoader(pileup_encoder, eval_vcf_loader, zyg_encoder, batch_size=128, shuffle=False, num_workers=32)
eval_vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
eval_vz_labels, eval_encoding = train_dataset()
eval_vz = alexnet(encoding=eval_encoding)
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
         optimization_params={"num_epochs": 1, "lr": 0.001},
         optimizer="adam")

######### Inference
#    # Load checkpointed model and run inference
#test_data_dir = get_data_folder()
#model_dir = os.path.join(test_data_dir, ".test_model")
#
## Create neural factory
#nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=model_dir)
#
## Generate dataset
#encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY, PileupEncoder.Layer.MAPPING_QUALITY]
#pileup_encoder = PileupEncoder(window_size = 100, max_reads = 100, layers = encoding_layers)
#bam = os.path.join(test_data_dir, "small_bam.bam")
#labels = os.path.join(test_data_dir, "candidates.vcf.gz")
#vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam=bam, is_fp=False)
#vcf_loader = VCFLabelLoader([vcf_bam_tuple])
#zyg_encoder = ZygosityLabelEncoder()
#test_dataset = VariantDataLoader(pileup_encoder, vcf_loader, zyg_encoder, batch_size=32, shuffle=False)
#
## Neural Network
#alexnet = AlexNet(num_input_channels=len(encoding_layers), num_vz=3)
#
## Create train DAG
#_, encoding = test_dataset()
#vz = alexnet(encoding=encoding)
#
## Invoke the "train" action.
#results = nf.infer([vz], checkpoint_dir=model_dir, verbose=True)
#for tensor_batches in results:
#    for batch in tensor_batches:
#        predicted_classes = torch.argmax(batch, dim=1)
#        for pred in predicted_classes:
#            print(zyg_encoder.decode_class(pred))
#
#shutil.rmtree(model_dir)
