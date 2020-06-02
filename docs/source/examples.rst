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

Training
````````

.. code-block:: python

    # Import nemo and variantworks modules
    import nemo
    from variantworks.dataloader import *
    from variantworks.io.vcfio import *
    from variantworks.networks import *
    from variantworks.sample_encoders import *

    # Create neural factory
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=tempdir)

    # Create pileup encoder by selecting layers to encode. More encoding layers
    # can be found in the documentation for PilupEncoder class.
    encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY]
    pileup_encoder = PileupEncoder(
        window_size=100, max_reads=100, layers=encoding_layers)

    # Instantiate a zygosity encoder that generates output labels. Converts a variant entry
    # into a class label for no variant, homozygous variant or heterozygous variant.
    zyg_encoder = ZygosityLabelEncoder()

    # Create neural network that receives 2 channel inputs (encoding layers defined above)
    # and outputs a logit over three classes (no variant, homozygous variant, heterozygous variant.
    model = AlexNet(num_input_channels=len(
        encoding_layers), num_output_logits=3)

    # Get datasets to train on. 
    # NOTE: To train a neural network well, the model needs to see samples from all types of classes.
    # The example here shows a file that has true variant (either homozygous or heterozygous),
    # but in practice one also needs to pass a set of false positive samples so the model can learn to
    # ignore them. False positive samples can be marked with `is_fp` so the reader can appripriately
    # assign their variant types.
    bam = os.path.join(get_data_folder(), "small_bam.bam")
    samples = os.path.join(get_data_folder(), "candidates.vcf.gz")
    vcf_loader = VCFReader([VCFReader.VcfBamPath(vcf=samples, bam=bam, is_fp=False)])

    # Create a data loader with custom sample and label encoder.
    dataset_train = ReadPileupDataLoader(ReadPileupDataLoader.Type.TRAIN, vcf_loader,
                                         batch_size=32, shuffle=True,
                                         sample_encoder=pileup_encoder, label_encoder=zyg_encoder)

    # Use CrossEntropyLoss to train.
    vz_ce_loss = nemo.backends.pytorch.common.losses.CrossEntropyLossNM(logits_ndim=2)

    # Create NeMo training DAG.
    vz_labels, encoding = dataset_train()
    vz = model(encoding=encoding)
    vz_loss = vz_ce_loss(logits=vz, labels=vz_labels)

    # Logger callback
    logger_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[vz_loss],
        print_func=lambda x: logging.info(f'Train Loss: {str(x[0].item())}'))
    )

    # Kick off training
    nf.train([vz_loss],
             callbacks=[logger_callback,
                        checkpoint_callback, evaluator_callback],
             optimization_params={"num_epochs": 4, "lr": 0.001},
             optimizer="adam")


Inference
`````````

The inference pipeline works in a very similar fashion, except the final NeMo DAG looks different.

.. code-block:: python

    # Import nemo and variantworks modules
    import nemo
    from variantworks.dataloader import *
    from variantworks.io.vcfio import *
    from variantworks.networks import *
    from variantworks.sample_encoders import *
    from variantworks.result_writer import *

    # Create neural factory. In this case, the checkpoint_dir has to be set for NeMo to pick
    # up a pre-trained model.
    nf = nemo.core.NeuralModuleFactory(
        placement=nemo.core.neural_factory.DeviceType.GPU, checkpoint_dir=model_dir)

    # Neural Network
    model = AlexNet(num_input_channels=len(
        encoding_layers), num_output_logits=3)

    # Dataset generation is done in a similar manner. It's important to note that the encoder used
    # for inference much match that for training.
    encoding_layers = [PileupEncoder.Layer.READ, PileupEncoder.Layer.BASE_QUALITY]
    pileup_encoder = PileupEncoder(
        window_size=100, max_reads=100, layers=encoding_layers)

    # Similar to training, a dataloader needs to be setup for the relevant datasets. In the case of
    # inference, it doesn't matter if the files are tagged as false positive or not. Each example will be
    # evaluated by the network. For simplicity the example is using the same dataset from training.
    # Note: No label encoder is required in inference.
    bam = os.path.join(test_data_dir, "small_bam.bam")
    labels = os.path.join(test_data_dir, "candidates.vcf.gz")
    vcf_bam_tuple = VCFReader.VcfBamPath(vcf=labels, bam=bam, is_fp=False)
    vcf_loader = VCFReader([vcf_bam_tuple])
    test_dataset = ReadPileupDataLoader(ReadPileupDataLoader.Type.TEST, vcf_loader, batch_size=32,
                                        shuffle=False, sample_encoder=pileup_encoder)

    # Create inference DAG
    encoding = test_dataset()
    vz = model(encoding=encoding)

    # Invoke the "infer" action.
    results = nf.infer([vz], checkpoint_dir=model_dir, verbose=True)

    # Instantiate a decoder that converts the predicted output of the network to
    # a zygosity enum.
    zyg_decoder = ZygosityLabelDecoder()

    # Decode inference results to labels
    for tensor_batches in results:
        for batch in tensor_batches:
            predicted_classes = torch.argmax(batch, dim=1)
            inferred_zygosity += [zyg_decoder(pred)
                                 for pred in predicted_classes]

    # Use the VCFResultWriter to output predicted zygosities to a VCF file.
    result_writer = VCFResultWriter(vcf_loader, inferred_zygosity)

    result_writer.write_output()
