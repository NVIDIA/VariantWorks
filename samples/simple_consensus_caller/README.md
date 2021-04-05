# Simple Deep Learning Consensus Sample

This sample shows how to setup a simple deep learning based pipeline
to learn and predict consensus sequence from reads.

For this sample, the following are necessary for each consensus region - 

1. Draft backbone.
2. Supporting reads.
2. Truth sequence.

## Third party software dependencies

The pre-processing of files done in this sample depend on several third-party
softwares. Please make sure you have them available in your environment in the
`$PATH` variable before continuing with this sample.

1. `minimap2` >= 2.17
2. `samtools` >= 1.10

## Data Generation

First step of this sample is to generate the encodings and labels used in training and evaluation.

The `pileup_hdf5_generator.py` script implements a data preparation script that leverages the
`SummaryEncoder` and `HaploidLabelEncoder` from VariantWorks to compute the feature and
label encodings, respectively. Generated encodings are then serialize into an HDF5 dataset.
This kind of pre-training data generation is common practice in neural network pipelines as
it yields more efficient usage of the GPU and overall better runtime performance.

### Input Data Folders
The `pileup_hdf5_generator.py` expects data to be organized in the following way for each sample when 
using `--data-dir/-d` or `--single-dir/-r` arguments:
```
-> example_folder
---> draft.fa
---> truth.fa
---> subreads.fa
```

The following is a sample command line for running the script.
```
python pileup_hdf5_generator.py -r `pwd`/data/samples/1 -o train.hdf -t 4
python pileup_hdf5_generator.py -r `pwd`/data/samples/2 -o eval.hdf -t 4
```
The `pileup_hdf5_generator.py` also has a `-d` option which can accept a folder with multiple
data examples underneath it, i.e. a folder of folders each of which is in the format described
above. This is added as a utility to enable processing of larger datasets.

To specify a different name for the draft_file, use the `--draft-name` to specify the name of the FASTA/FASTQ file containing the draft sequence.

If the base quality score of the draft is to be extracted and encoded, please add the `--use-quality` flag.

### Input BAM Files
Alternatively, the `pileup_hdf5_generator.py` script also supports BAM files as input by providing the BAM file path for the
subreads data (`--subreads-file`) and the drafts data (`--draft-file`) along with the reference genome file path (`--reference`) in FASTA format.
```
python pileup_hdf5_generator.py --draft-file <path to draft_file.bam> --subreads-file <path to subreads_file.bam> --reference <path to reference.fa> -o train.hdf -t 4
```

The expected format of the QNAME field of every read in the subreads BAM file is:

```<subreads dataset id>/<molecule id>/<start position>_<end position>```
e.g. ```m54238_180903_015530/4194990/6704_19829```

Similarly, For the drafts BAM file the format of the QNAME filed must be:

```<subreads dataset id>/<molecule id>/<string>```, for example  ```m54238_180903_015530/4456953/<string>```

The entries in the subreads & drafts BAM files have to be sorted from the lower molecule id number to the higher molecule id before executing the above-mentioned script.

To extract base quality scores of the draft add `--use-quality` flag to the pileup generation command line.

For more details on the script usage, please refer to its help message.
```
python pileup_hdf5_generator.py -h
```

## Training pipeline

Once the dataset has been generated, the next step is to plug them into the trainer.

The trainer is a deep learning training and evaluation pipeline configured using the
[NeMo](https://nvidia.github.io/NeMo/) API. The key VariantWorks components highlighted in
the trainer are the `HDFDataLoader` class (from `variantworks.dataloader`) that serves samples
and labels to the network during training and evaluation, and a reference RNN network architecture
, `ConsensusRNN`, suitable for consensus calling (from `variantworks.networks`).

The following is a sample command line for the trainer.
```
python consensus_trainer.py --train-hdf train.hdf --epochs 50 --model-dir `pwd`/sample_models --eval-hdf eval.hdf
```

Details of the script usage can be found in its help message.
```
python consensus_trainer.py -h
```

The training script provides options to configure the model architecture as well. The `--model` flag can be used to select 
RNN or CNN architectures. If a model needs to be trained on encodings that include base quality scores and reference 
nucleotide information, then the input size of the RNN needs to match the dimensions of the encoding. So the following 
command line can be used to address that (assuming the input size is 16 channels) -
```
python consensus_trainer.py --train-hdf train.hdf --epochs 50 --model-dir `pwd`/sample_models --eval-hdf eval.hdf --input_feature_size 16
```

### Multi-GPU training
The consensus caller trainer uses the built-in multi GPU training offered by `NeMo`. To run training over multiple GPUs,
add the following options before running the trainer
```
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> consensus_trainer.py ...
```

## Inference pipeline

The inference script takes care of running a pre-trained model on unseen data, post-processesing
the model output into a genomic sequence, and writing the sequence out to a `fasta` file. For each of
these steps, the inference script uses componets from the VariantWorks API such as the `HDFDataLoader`,
the reference network architecture and the concatenation utility for merging model outputs into
the final consensus sequence.

The current inference script is setup to only processes a single consensus sequence at a time,
i.e. the hdf file passed to it must contain encodings from a single underlying sequence. If multiple
sequence encodings are present, the script may concatenate them into a single incorrect sequence.

Here's a sample command line for the inference script using a pre-trained model.
```
python pileup_hdf5_generator.py -r `pwd`/data/samples/3 -o infer.hdf
python consensus_infer.py --infer-hdf infer.hdf --model-dir `pwd`/sample_models -o sample.fasta
```
