# Simple SNP Caller Sample

This sample shows how to setup a simple SNP zygosity caller training and inference pipeline to learn
and predict zygosity for each variant sample.

## Training/Inference

For this, the following are necessary - 

1. BAM file with read information.
2. True and False positive VCF files to help teach the network the difference between
real and false variants.

The sample trainer and inferer makes use of the VCFReader class to load and parse input VCF files.
The samples are encoded by the PileupEncoder and ZygosityLabelEncoder classes, which are
encapsualted in ReadPileupDataLoader. What this structure implies is that the encoding for each
sample is generated online during training. This type of repeated, online data generation is not 
the recommended approach for scalable neural network pipelines, and is only implemented in this manner
for instructional purposes.

Note: For production pipelines, encodings can be generated offline and saved in
a binary format such as HDF5 (see the hdf5-pileup-generator example for that), which can
speed up overall application runtime as the encodings don't need to be generated on the fly.

With the data layer implemented, the rest of the neural network pipeline is
programmed using the [NeMo](https://nvidia.github.io/NeMo/) API.

The trainer is setup to serialize the network after epoch. The serialized model
can be used to run inference after a full training loop.

Once inference is complete, the predicted zygosities are written out to another VCF
file using the VCFResultWriter API from VariantWorks.

Details of the script usage can be found in it's help message.
```
./cnn_snp_caller -h
```
