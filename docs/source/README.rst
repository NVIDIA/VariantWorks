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

Getting Started
---------------

#. Run the following command to install the required libraries:

    .. code-block:: bash

        apt install zlib1g-dev libcurl4-gnutls-dev libssl-dev libbz2-dev liblzma-dev

#. `htslib` is required and can be installed from source by following the instructions in: https://github.com/samtools/htslib#building-htslib


#. Install latest development code from source:

    .. code-block:: bash

        git clone https://github.com/clara-parabricks/VariantWorks.git
        cd VariantWorks
        pip install -r requirements-python-style.txt
        pip install -r requirements.txt
        pip install -e .
        # Install pre-push hooks to run tests
        ln -nfs $(readlink -f hooks/pre-push) .git/hooks/pre-push

Alternative Anaconda Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Clone VariantWorks repository and change directory:

    .. code-block:: bash

        git clone https://github.com/clara-parabricks/VariantWorks.git
        cd VariantWorks


#. Create & activate a new environment with the required OS libraries:

    .. code-block:: bash

        conda env create --name <ENVIRONMENT_NAME> -f ./environment.yml
        conda activate <ENVIRONMENT_NAME>

#. Some Python packages depends on the headers and shared libraries which were installed in the previous step,
   therefore before executing `pip`, run the following command:

    .. code-block:: bash

        export LDFLAGS=-L<ENVIRONMENT_PATH>/lib
        export CPPFLAGS=-I<ENVIRONMENT_PATH>/include

   The environment path can be queried by running:

    .. code-block:: bash

        conda info --envs

#. Install Python requirements and VariantWorks package from source

    .. code-block:: bash

        pip install -r requirements-python-style.txt
        pip install -r requirements.txt
        pip install -e .
        # Install pre-push hooks to run tests
        ln -nfs $(readlink -f hooks/pre-push) .git/hooks/pre-push