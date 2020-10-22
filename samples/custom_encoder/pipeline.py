#!/usr/bin/env/python3


import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import eval_epochs_done_callback, eval_iter_callback
import torch

from variantworks.encoders import Encoder
from variantworks.dataloader import ReadPileupDataLoader
from variantworks.io.vcfio import VCFReader, VCFWriter
from variantworks.networks import MLP

class CategoricalAccuracy(object):
    """Categorical accuracy metric."""

    def __init__(self):
        """Constructor for metric."""
        self._tp = 0
        self._fn = 0
        self._num_examples = 0
        self._fp = 0

    def __call__(self, y_true, y_pred):
        """Compute categorical accuracy."""
        preds = torch.argmax(y_pred, dim=1)
        #correct = torch.eq(preds, y_true).view(-1)
        correct = preds + y_true
        correct = correct > 1
        #print(y_true)
        #print(preds)
        #print(correct)
        tp = torch.sum(correct).item()
        fn = (torch.sum(y_true).item() - tp)
        fp = (torch.sum(preds).item() - tp)
        self._tp += tp
        self._fn += fn
        self._fp += fp

        if self._tp + self._fp == 0:
            prec = 1.0
        else:
            prec = (self._tp / (self._tp + self._fp))

        if self._tp + self._fn == 0:
            rec = 1.0
        else:
            rec = (self._tp / (self._tp + self._fn))

        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)

        self._num_examples += correct.shape[0]
        print(self._tp, self._fp, self._fn)
        return f1

    def reset(self):
        #print("reset")
        self._tp = 0
        self._fp = 0
        self._fn = 0


def generate_eval_callback(categorical_accuracy_func):
    """Custom callback function generator for network evaluation loop."""
    def eval_iter_callback(tensors, global_vars):
        if "eval_loss" not in global_vars.keys():
            global_vars["eval_loss"] = []
        for kv, v in tensors.items():
            if kv.startswith("loss"):
                global_vars["eval_loss"].append(torch.mean(torch.stack(v)).item())
                # global_vars['eval_loss'].append(v.item())

        if "top1" not in global_vars.keys():
            global_vars["top1"] = []
            categorical_accuracy_func.reset()

        output = None
        labels = None
        for kv, v in tensors.items():
            if kv.startswith("output"):
                # output = tensors[kv]
                output = torch.cat(tensors[kv])
            if kv.startswith("label"):
                # labels = tensors[kv]
                labels = torch.cat(tensors[kv])

        if output is None:
            raise Exception("output is None")

        with torch.no_grad():
            accuracy = categorical_accuracy_func(labels, output)
            global_vars["top1"].append(accuracy)

    return eval_iter_callback


class CustomEncoder(Encoder):
    def __init__(self, features=[], filters=[]):
        self._features = features
        self._filters = filters

    def __call__(self, variant):
        data = []
        for key in self._features:
            data.append(variant.info[key])
        for k in self._filters:
            data.append(1. if (variant.filter is not None and k in variant.filter) else 0.)
        tensor = torch.FloatTensor(data)
        return tensor

class CustomLabeler(Encoder):
    def __init__(self, truth_col):
        self._truth_col = truth_col

    def __call__(self, variant):
        return variant.info[self._truth_col]


def train():
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU)

    features = ["strelka_x", "strelka_y", "mutect", "varscan_x", "varscan_y", "sniper"]
    filters = ["PASS_x_1", "PASS_x", "PASS_y"]

    model = MLP(len(features) + len(filters), 2, 1000)

    train_vcf_1 = VCFReader("/home/jdaw/tijyojwad/somatic-vc/tools/set3_labeled.vcf",
                            info_keys=["*"],
                            filter_keys=[],
                            format_keys=[],
                            require_genotype=False)
    #train_vcf_2 = VCFReader("/home/jdaw/tijyojwad/somatic-vc/tools/set2_labeled.vcf",
    #                        info_keys=["*"],
    #                        filter_keys=["*"],
    #                        format_keys=["*"],
    #                        require_genotype=False)
    #train_vcf_3 = VCFReader("/home/jdaw/tijyojwad/somatic-vc/tools/small.vcf",
    #                        info_keys=["*"],
    #                        filter_keys=[],
    #                        format_keys=[],
    #                        require_genotype=False)

    encoder = CustomEncoder(features=features, filters=filters)

    train_dataset = ReadPileupDataLoader(ReadPileupDataLoader.Type.TRAIN,
                                         [train_vcf_1],# train_vcf_2],
                                         batch_size=32, shuffle=True,
                                         sample_encoder=encoder,
                                         label_encoder=CustomLabeler("truth"),
                                         num_workers=32)
    vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    labels, encoding = train_dataset()
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

    eval_dataset = ReadPileupDataLoader(ReadPileupDataLoader.Type.EVAL,
                                         [train_vcf_1],
                                         batch_size=(8192*4), shuffle=False,
                                         sample_encoder=encoder,
                                         label_encoder=CustomLabeler("truth"),
                                         num_workers=32)
    eval_vz_ce_loss = CrossEntropyLossNM(logits_ndim=2)
    eval_vz_labels, eval_encoding= eval_dataset()
    eval_vz = model(encoding=eval_encoding)
    eval_vz_loss = eval_vz_ce_loss(logits=eval_vz, labels=eval_vz_labels)

    # Add evaluation callback
    evaluator_callback = nemo.core.EvaluatorCallback(
        eval_tensors=[eval_vz_loss, eval_vz, eval_vz_labels],
        #user_iter_callback=eval_iter_callback,
        user_iter_callback=generate_eval_callback(CategoricalAccuracy()),
        user_epochs_done_callback=eval_epochs_done_callback,
        eval_step=15000,
        eval_epoch=1,
        eval_at_start=True,
    )
    callbacks.append(evaluator_callback)

    # Invoke the "train" action.
    nf.train([vz_loss],
             callbacks=callbacks,
             optimization_params={"num_epochs": 100, "lr": 0.00001},
             optimizer="adam")

train()
