.. VariantWorks SDK documentation master file, created by
   sphinx-quickstart on Mon Jun  1 21:18:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Examples
========


VariantWorks can be used to setup training and inference pipelines using the available encoders, data loaders
and I/O libraries.

SNP Zygosity Predictor
----------------------

This example showcases how to develop a SNP zygosity predictor from variant SNP candidates. This example is
also available as a sample.

For more details on training and inference using NeMo (including features such as multi-GPU training, half-precision
trainig, model parallel training, etc.), please refer to the `Nemo documentation <https://nvidia.github.io/NeMo/tutorials/examples.html>`_.

Training
````````
.. literalinclude:: ./snippets/snp_zygosity_predictor_training.py
   :language: python
   :caption: snp_zygosity_predictor_training.py
   :lines: 19-

Inference
`````````
The inference pipeline works in a very similar fashion, except the final NeMo DAG looks different.

.. literalinclude:: ./snippets/snp_zygosity_predictor_inference.py
   :language: python
   :caption: snp_zygosity_predictor_inference.py
   :lines: 19-


HDF5 Pileup Dataset Generator
-----------------------------

This example is designed to highlight how the encoder classes can be used independent
of the training framework to generate encodings for samples and serializing them to a
file for later consumption. This sort of pipeline is often used when data generation
takes a proportionally larger portion of the compute time compared to the network training
or inference components.

.. literalinclude:: ./snippets/hdf5_pileup_dataset_generator.py
   :language: python
   :caption: hdf5_pileup_dataset_generator.py
   :lines: 19-
