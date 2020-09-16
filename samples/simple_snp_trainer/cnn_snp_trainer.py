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

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import eval_epochs_done_callback, eval_iter_callback

from variantworks.dataloader import HDFPileupDataLoader
from variantworks.networks import AlexNet
from variantworks.neural_types import ReadPileupNeuralType, VariantZygosityNeuralType


def create_model():
    """Return neural network to train."""
    # Neural Network
    alexnet = AlexNet(num_input_channels=2, num_output_logits=3)

    return alexnet


def train(args):
    """Train a sample model with test data."""
    # Create neural factory as per NeMo requirements.
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU)

    model = create_model()

    # Create train DAG
    train_dataset = HDFPileupDataLoader(HDFPileupDataLoader.Type.TRAIN, args.train_hdf, batch_size=32,
                                        shuffle=True, num_workers=args.threads,
                                        hdf_encoding_key="encodings", hdf_label_key="labels",
                                        encoding_dims=('B', 'C', 'H', 'W'), label_dims=tuple('B'),
                                        encoding_neural_type=ReadPileupNeuralType(),
                                        label_neural_type=VariantZygosityNeuralType())
    vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    vz_labels, encoding = train_dataset()
    vz = model(encoding=encoding)
    vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)

    callbacks = []

    # Logger callback
    loggercallback = nemo.core.SimpleLossLoggerCallback(
        tensors=[vz_loss],
        step_freq=5,
        print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'),
    )
    callbacks.append(loggercallback)

    # Checkpointing models through NeMo callback
    checkpointcallback = nemo.core.CheckpointCallback(
        folder=args.model_dir,
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

    # Create eval DAG if eval files are available
    if args.eval_hdf:
        eval_dataset = HDFPileupDataLoader(HDFPileupDataLoader.Type.EVAL, args.eval_hdf, batch_size=32,
                                           shuffle=False, num_workers=args.threads,
                                           hdf_encoding_key="encodings", hdf_label_key="labels",
                                           encoding_dims=('B', 'C', 'H', 'W'), label_dims=tuple('B'),
                                           encoding_neural_type=ReadPileupNeuralType(),
                                           label_neural_type=VariantZygosityNeuralType())
        eval_vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
        eval_vz_labels, eval_encoding = eval_dataset()
        eval_vz = model(encoding=eval_encoding)
        eval_vz_loss = eval_vz_ce_loss(logits=eval_vz, labels=eval_vz_labels)

        # Add evaluation callback
        evaluator_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[eval_vz_loss, eval_vz, eval_vz_labels],
            user_iter_callback=eval_iter_callback,
            user_epochs_done_callback=eval_epochs_done_callback,
            eval_step=100,
            eval_epoch=1,
            eval_at_start=False,
        )
        callbacks.append(evaluator_callback)

    # Invoke the "train" action.
    nf.train([vz_loss],
             callbacks=callbacks,
             optimization_params={"num_epochs": args.epochs, "lr": 0.001},
             optimizer="adam")


def build_parser():
    """Build parser object with options for sample."""
    parser = argparse.ArgumentParser(
        description="Simple SNP caller based on VariantWorks.")

    parser.add_argument("--train_hdf",
                        help="HDF with examples for training.",
                        required=True)
    parser.add_argument("--eval_hdf",
                        help="HDF with examples for evaluation.",
                        required=False)
    parser.add_argument("--epochs", type=int,
                        help="Epochs for training.",
                        required=False, default=1)
    import multiprocessing
    parser.add_argument("-t", "--threads", type=int,
                        help="Threads to use for parallel loading.",
                        required=False, default=multiprocessing.cpu_count())
    parser.add_argument("--model_dir", type=str,
                        help="Directory for storing trained model checkpoints. Stored after every eppoch of training.",
                        required=False, default="./models")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
