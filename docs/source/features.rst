.. VariantWorks SDK documentation master file, created by
   sphinx-quickstart on Mon Jun  1 21:18:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Core Features
=============


VariantWorks provides most of its functionality in the form of library functions that
encapsulate common algorithms, encodings and utilites helpful for variant processing.
The following sections will walk through an explanation of each feature that is currently
supported.

Encoders
--------

Encoders are classes that generate an encoding for the neural network. The classes can be used to
not only generate encodings, but also augment and transform them.

All encoders need to inherit from the base class :class:`Encoder<variantworks.encoders.Encoder>`
and implement the abstract method `__call__`, which triggers the encoding generation.

.. image:: encoder.png

Currently available encoders -

* :class:`PileupEncoder<variantworks.encoders.PileupEncoder>` - Encoding read pileups as multi dimensional images.
* :class:`ZygosityLabelEncoder<variantworks.encoders.ZygosityLabelEncoder>` - Encoding a variant entry into a zygosity label.
* :class:`SummaryEncoder<variantworks.encoders.SummaryEncoder>` - Encoding pileups into a matrix of nucleotide summary counts.
* :class:`HaploidLabelEncoder<variantworks.encoders.HaploidLabelEncoder>` - Encoding pileups into labels with true nucleotide sequence.

Encoders are also encouraged to implement a visualization function to enable visual inspection of
encodings for debugging and analysis. This is optional for encoders, and is currently only implemented
for :meth:`PileupEncoder.visualize<variantworks.encoders.PileupEncoder.visualize>`.

I/O
---

Format Readers
``````````````

Reading entries from common genomics file formats (such as BAM, VCF) is a precursor to almost all variant calling pipelines.
VariantWorks provides parsing (based on cyvcf2) for VCFs, and leverages pySAM for working with BAM files.

The Reader classes follow a similar structure to Encoders when it comes to inheritence. All readers must inherit from
:class:`BaseReader<variantworks.io.baseio.BaseReader>` and implement the abstract methods `__len__` and `__getitem__` to make them
compatible and easy to use with data loaders.

Currently available readers - 

* VCF I/O module provides :class:`VCFReader<variantworks.io.vcfio.VCFReader>` and :class:`VCFWriter<variantworks.io.vcfio.VCFWriter>` for
  parsing VCFs into dataframes and serializing VCF dataframes back into files.


Data Loaders
````````````

Data loaders are core to generating datasets in batches for training a nerual network. Since VariantWorks is based on the
NeMo toolkit, we leverage the data loader abstractions defined in 
`DataLayerNM <https://nvidia.github.io/NeMo/tutorials/custommodules.html#data-layer-module>`_.

Currently available data loaders - 

* :class:`ReadPileupDataLoader<variantworks.dataloader.ReadPileupDataLoader>` - encapsulates loading samples from VCF and using PileupEncoders to generate training data.
  This type of data loader is typically useful for variant calling tasks which process BAMs and VCFs simultaneously.
* :class:`HDFDataLoader<variantworks.dataloader.HDFDataLoader>` - encapsulates a generalized, multi-threaded data loader for loading tensors from HDF files. This type
  of data loader is frequently used when data is prepared/serialized ahead of time into a HDF file and directly read from the HDF file during training/evaluation loops.


Reference Networks
------------------

Reference architectures suitable for variant calling and/or consensus calling are packaged
as part of VariantWorks to quickly bootstrap DL powered variant caling pipelies.

Like the data loaders, reference neural networks are also defined within the NeMo paradigm as
`TrainableNM <https://nvidia.github.io/NeMo/tutorials/custommodules.html#trainable-module>`_ modules.

Currently available networks -

* :class:`AlexNext<variantworks.networks.AlexNet>`
* :class:`AlexNext<variantworks.networks.ConsensusRNN>`
