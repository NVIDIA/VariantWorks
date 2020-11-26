# Custom Encoder Samples

This sample covers how to write a custom encoder in `VariantWorks` using scalar
values from a VCF. Such a setup is useful when considering pipelines which may
require additional metadata information to be passed into the network along with, e.g.
the read pileup, to better inform the network about the context.

## Training pipeline

The training pipeline in this sample is kept extremely simple. The training and evaluation data
is generated using an online encoder (i.e. an encoder that gnerates tensors during the training pipeline)
and passed into a multi-layer perceptron. The objective of this pipeline is to predict the zygosity of
each variant only using metadata from the variant records.

Key files in this sample are:

1. `custom_encoder.py` - This file contains the custom encoder implementation for the VCF scalar values.
2. `custom_model.py` - This file contains a simple MLP model.
3. `pipeline.py` - The training and evaluation pipeline.

## Data

A sample training and evaluation VCF is available under the `data` folder. This can be used to run the sample.
