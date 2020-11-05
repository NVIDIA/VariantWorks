from variantworks.networks import ConsensusRNN

def create_model(args):
    """Return neural network to train."""
    # Neural Network
    rnn = ConsensusRNN(sequence_length=args.sequence_length,
                       input_feature_size=args.input_feature_size,
                       num_output_logits=args.num_output_logits,
                       gru_size = args.gru_size,
                       gru_layers = args.gru_layers,
                       apply_softmax=True)

    return rnn

