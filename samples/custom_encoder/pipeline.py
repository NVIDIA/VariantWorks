#!/usr/bin/env/python3

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
"""A sample program showcasing the use of custom encoders based on scalar VCF keys."""

import nemo
from nemo import logging
from nemo.core.neural_types import VoidType, LabelsType
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import eval_epochs_done_callback, eval_iter_callback
import os

from variantworks.encoders import ZygosityLabelEncoder
from variantworks.dataloader import VariantDataLoader
from variantworks.io.vcfio import VCFReader

from custom_encoder import CustomEncoder
from custom_model import MLP


def main():
    """Training function."""
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU)

    # Scalar features to use from VCF
    format_keys = ["AD", "ADALL", "DP"]

    # Create a model. AD and ADALL are "R" header type, so each
    # record has 2 values for each - one for reference and once
    # for alt allele.
    model = MLP(5, 5, 3)

    cwd = os.path.dirname(os.path.realpath(__file__))
    train_vcf = VCFReader(os.path.join(cwd, "data", "train.vcf"), format_keys=["*"])
    eval_vcf = VCFReader(os.path.join("data", "eval.vcf"), format_keys=["*"])

    encoder = CustomEncoder(vcf_format_keys=format_keys)

    train_dataset = VariantDataLoader(VariantDataLoader.Type.TRAIN,
                                      [train_vcf],
                                      batch_size=32, shuffle=True,
                                      input_encoder=encoder,
                                      encoder_dims=[('B', 'D')],
                                      encoder_neural_types=[VoidType()],
                                      label_encoder=ZygosityLabelEncoder(),
                                      label_dims=[('B')],
                                      label_neural_types=[LabelsType()],
                                      num_workers=32)

    vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    encoding, labels = train_dataset()
    pred = model(encoding=encoding)
    vz_loss = vz_ce_loss(logits=pred, labels=labels)

    callbacks = []

    # Logger callback
    loggercallback = nemo.core.SimpleLossLoggerCallback(
        tensors=[vz_loss],
        step_freq=500,
        print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
    )
    callbacks.append(loggercallback)

    # Checkpointing models through NeMo callback
    checkpointcallback = nemo.core.CheckpointCallback(
        folder="./temp-model",
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
    callbacks.append(checkpointcallback)

    eval_dataset = VariantDataLoader(VariantDataLoader.Type.EVAL,
                                     [eval_vcf],
                                     batch_size=32, shuffle=True,
                                     input_encoder=encoder,
                                     encoder_dims=[('B', 'D')],
                                     encoder_neural_types=[VoidType()],
                                     label_encoder=ZygosityLabelEncoder(),
                                     label_dims=[('B')],
                                     label_neural_types=[LabelsType()],
                                     num_workers=32)

    eval_vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    eval_encoding, eval_vz_labels = eval_dataset()
    eval_vz = model(encoding=eval_encoding)
    eval_vz_loss = eval_vz_ce_loss(logits=eval_vz, labels=eval_vz_labels)

    # Add evaluation callback
    evaluator_callback = nemo.core.EvaluatorCallback(
        eval_tensors=[eval_vz_loss, eval_vz, eval_vz_labels],
        user_iter_callback=eval_iter_callback,
        user_epochs_done_callback=eval_epochs_done_callback,
        eval_step=5000,
        eval_epoch=1,
        eval_at_start=True,
    )
    callbacks.append(evaluator_callback)

    # Invoke the "train" action.
    nf.train([vz_loss],
             callbacks=callbacks,
             optimization_params={"num_epochs": 100, "lr": 0.0001},
             optimizer="adam")


if __name__ == "__main__":
    main()
