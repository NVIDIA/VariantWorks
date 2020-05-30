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

from distutils.dir_util import copy_tree
import os
import pytest
import shutil
import tempfile
import torch
import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import compute_accuracy, eval_epochs_done_callback, eval_iter_callback

from claragenomics.variantworks.dataset import VariantDataLoader
from claragenomics.variantworks.label_loader import VCFLabelLoader
from claragenomics.variantworks.networks import AlexNet
from claragenomics.variantworks.result_writer import VCFResultWriter
from claragenomics.variantworks.variant_encoder import PileupEncoder, ZygosityLabelEncoder


from test_utils import get_data_folder


def test_simple_vc_trainer():
    # Train a sample model with test data

    # Create temporary folder
    tempdir = tempfile.mkdtemp()

    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=tempdir)

    # Generate dataset
    encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY, PileupEncoder.Layer.MAPPING_QUALITY,
                       PileupEncoder.Layer.REFERENCE, PileupEncoder.Layer.ALLELE]
    pileup_encoder = PileupEncoder(window_size=100, max_reads=100, layers=encoding_layers)
    bam = os.path.join(get_data_folder(), "small_bam.bam")
    labels = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam=bam, is_fp=False)
    vcf_loader = VCFLabelLoader([vcf_bam_tuple])
    zyg_encoder = ZygosityLabelEncoder()

    # Neural Network
    alexnet = AlexNet(num_input_channels=len(encoding_layers), num_vz=3)

    # Create train DAG
    dataset_train = VariantDataLoader(pileup_encoder, vcf_loader, zyg_encoder, batch_size=32, shuffle=True)
    vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    vz_labels, encoding = dataset_train()
    vz = alexnet(encoding=encoding)
    vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)

    # Create evaluation DAG using same dataset as training
    dataset_eval = VariantDataLoader(pileup_encoder, vcf_loader, zyg_encoder, batch_size=32, shuffle=False)
    vz_ce_loss_eval = CrossEntropyLossNM(logits_ndim=2)
    vz_labels_eval, encoding_eval = dataset_eval()
    vz_eval = alexnet(encoding=encoding_eval)
    vz_loss_eval = vz_ce_loss_eval(logits=vz_eval, labels=vz_labels_eval)

    # Logger callback
    logger_callback = nemo.core.SimpleLossLoggerCallback(
            tensors=[vz_loss, vz, vz_labels],
            step_freq=1,
            )

    evaluator_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[vz_loss_eval, vz_eval, vz_labels_eval],
            user_iter_callback=eval_iter_callback,
            user_epochs_done_callback=eval_epochs_done_callback,
            eval_step=1,
            )

    # Checkpointing models through NeMo callback
    checkpoint_callback = nemo.core.CheckpointCallback(
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

    # Invoke the "train" action.
    nf.train([vz_loss],
            callbacks=[logger_callback, checkpoint_callback, evaluator_callback],
            optimization_params={"num_epochs": 4, "lr": 0.001},
            optimizer="adam")

    # Remove checkpoint directory
    model_dir = os.path.join(get_data_folder(), ".test_model")
    copy_tree(tempdir, model_dir)
    shutil.rmtree(tempdir)


@pytest.mark.depends(on=['test_simple_vc_trainer'])
def test_simple_vc_infer():
    # Load checkpointed model and run inference
    test_data_dir = get_data_folder()
    model_dir = os.path.join(test_data_dir, ".test_model")

    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=model_dir)

    # Generate dataset
    encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY, PileupEncoder.Layer.MAPPING_QUALITY,
                       PileupEncoder.Layer.REFERENCE, PileupEncoder.Layer.ALLELE]
    pileup_encoder = PileupEncoder(window_size=100, max_reads=100, layers=encoding_layers)
    bam = os.path.join(test_data_dir, "small_bam.bam")
    labels = os.path.join(test_data_dir, "candidates.vcf.gz")
    vcf_bam_tuple = VCFLabelLoader.VcfBamPaths(vcf=labels, bam=bam, is_fp=False)
    vcf_loader = VCFLabelLoader([vcf_bam_tuple])
    zyg_encoder = ZygosityLabelEncoder()
    test_dataset = VariantDataLoader(pileup_encoder, vcf_loader, zyg_encoder, batch_size=32, shuffle=False)

    # Neural Network
    alexnet = AlexNet(num_input_channels=len(encoding_layers), num_vz=3)

    # Create train DAG
    _, encoding = test_dataset()
    vz = alexnet(encoding=encoding)

    # Invoke the "train" action.
    results = nf.infer([vz], checkpoint_dir=model_dir, verbose=True)

    # Decode inference results to labels
    for tensor_batches in results:
        for batch in tensor_batches:
            predicted_classes = torch.argmax(batch, dim=1)
            inferred_zygosity = [zyg_encoder.decode_class(pred) for pred in predicted_classes]

    result_writer = VCFResultWriter(vcf_loader, inferred_zygosity)

    result_writer.write_output()

    shutil.rmtree(model_dir)
