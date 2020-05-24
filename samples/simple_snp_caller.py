import os
import shutil
import tempfile
import time

import nemo
from nemo import logging
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM
from nemo.backends.pytorch.torchvision.helpers import compute_accuracy
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
tp_vcf = VCFLabelLoader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/tp_vcf_1m_giab.vcf.gz", bam=bam, is_fp=False)
fp_vcf = VCFLabelLoader.VcfBamPaths(vcf="/ssd/VariantWorks/end_to_end_workflow_sample_files/train_fp.vcf.gz", bam=bam, is_fp=True)
vcf_loader = VCFLabelLoader([tp_vcf, fp_vcf])
print(len(vcf_loader))
label_loader_time = time.time()
print("Label loading time {}".format(label_loader_time - start_time))
zyg_encoder = ZygosityLabelEncoder()
train_dataset = VariantDataLoader(pileup_encoder, vcf_loader, zyg_encoder, batch_size=32, shuffle=False, num_workers=32)

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

# Logger callback
loggercallback = nemo.core.SimpleLossLoggerCallback(
        tensors=[vz_loss, vz, vz_labels],
        print_func=my_print_fn,
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

# Invoke the "train" action.
nf.train([vz_loss],
         callbacks=[loggercallback, checkpointcallback],
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
