VariantWorks SDK
================

VariantWorks is a framework to enable the development of Deep Learning based genomic read processing tasks such as
variant calling, consensus calling, etc. It provides a library of data encoding and parsing functions commonly
applicable to read processing, along with a simple way to plug them into a Deep Learning pipeline.

For the Deep Learning pipeline, VariantWorks leverages the `NeMo <https://nvidia.github.io/NeMo/>`_ framework
which provdes an easy-to-use, graph based representation of high level computation graphs.

The target audience for VariantWorks is the following -

#. `Variant Caller developers` - for existing developers in the variant calling community, VariantWorks
   intends to provide a convenient way to start designing variant callers built using Deep Learning.
#. `Deep Learning practitioners` - for existing deep learning practitioners, VariantWorks can lower the barrier
   to applying novel Deep Learning techniques to the field of genomic variant calling.

Core Features
-------------

* Encoders - Pre-written, commonly used (and in the future, optimized) encoders for reads.
* I/O - Readers and writers for common genomics file formats.
* Reference Models - Collection of neural network architectures well suited for variant calling.

Requirements
------------

#. Python 3.7+
#. NVIDIA GPU (Pascal+ architecture)
#. `NVIDIA Apex <https://github.com/nvidia/apex>`__ library (for multi-GPU training in supported pipelines)

Getting Started
---------------

* Install latest development code from source

.. code-block:: bash

    git clone --recursive https://github.com/clara-parabricks/VariantWorks.git
    cd VariantWorks
    pip install -r python-style-requirements.txt
    pip install -r requirements.txt
    pip install -e .
    # Install pre-push hooks to run tests
    ln -nfs $(readlink -f hooks/pre-push) .git/hooks/pre-push

Sample Data
---------------
We provide sample data to explore the sample scripts for the *simple_snp_trainer*:
The data was derived from https://github.com/clara-parabricks/DL4VC/blob/master/docs/Data.md after generating
variant candidates and then intersect them with known truth set for variants in that region.

* Bam files
    #. Chr 1
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.chr1.bam
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.chr1.bam.bai
    #. Chr 10
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.chr10.bam
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.chr10.bam.bai
    #. Chr 17
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.chr17.bam
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878-50x.sort.chr17.bam.bai

* VCF files:
    #. Chr 1
        *  https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_TP_chr1.vcf (True Positive)
        *  https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_FP_chr1.vcf (False Positive)
    #. Chr1 (subset, first 7000 variants)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_TP_chr1_7000samples.vcf (True Positive)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_FP_chr1_7000samples.vcf (False Positive)
    #. Chr 10
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_TP_chr10.vcf (True Positive)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_FP_chr10.vcf (False Positive)
    #. Chr10 (subset, first 7000 variants)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_TP_chr10_7000samples.vcf (True Positive)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_FP_chr10_7000samples.vcf (False Positive)
    #. Chr 17
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_TP_chr17.vcf (True Positive)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_FP_chr17.vcf (False Positive)
    #. Chr17 (subset, first 7000 variants)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_TP_chr17_7000samples.vcf (True Positive)
        * https://variantworks.s3.us-east-2.amazonaws.com/HG001-NA12878_FP_chr17_7000samples.vcf (False Positive)
