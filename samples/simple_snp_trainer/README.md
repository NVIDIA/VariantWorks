# Simple SNP Trainer Sample

This sample shows how to setup a simple SNP zygosity caller training pipeline to learn
and predict zygosity for each variant sample.

For this sample, the following are necessary - 

1. BAM file with read information.
2. True and False positive VCF files to help teach the network the difference between
real and false variants.

## Data Generation

First step of this sample is to generate the encodings and labels used in training and evaluation.

The `pileup_hdf5_generator.py` script showcases ways to leverage the various encoder APIs in VariantWorks
to compute the encodings and serialize into an HDF5 dataset. This kind of pre-training data generation
is common practice in neural network pipelines as it yields more efficient usage of the GPU and hence overall
better runtime performance.

`pileup_hdf5_generator.py` in particular highlights the use of `VCReader` to load and parse input VCF files,
followed by the `PileupEncoder` and `ZygosityLabelEncoder` APIs for the encoding itself.

For details on the script usage, please refer to its help message.
```
pileup_hdf5_generator.py -h
```

## Training pipeline

Once the dataset has been generated, the next step is to plug them into the trainer.

Majory of the trainer is a training pipeline programmed using the the [NeMo](https://nvidia.github.io/NeMo/) API.
The key VariantWorks API highlighted in the trainer is the `HDFPileupDataLoader` class that serves
samples and labels to the network during training.

Details of the script usage can be found in it's help message.
```
./cnn_snp_trainer.py -h
```
