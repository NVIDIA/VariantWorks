# Abstract class for creating a dataset from BAM and VCF files

from torch.utils.data import Dataset, DataLoader

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import ChannelType, LabelsType, LossType, NeuralType

class SnpPileupDataType(DataLayerNM):
    """
    Data layer that outputs (pileup, variant type) data and label pairs.

    Args:
        bam : Path to BAM file
        labels : Path to labels file
        pileup_generator : Callable class defining pileup generator
        batch_size : batch size for dataset [32]
        shuffle : shuffle dataset [True]
        num_workers : numbers of parallel data loader threads [4]
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports
        """
        return {
            "pileup": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
            "label": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(self, bam, labels, pileup_generator, batch_size=32, shuffle=True, num_workers=4):
        super().__init__()

        class DatasetWrapper(Dataset):
            def __init__(self, bam, labels, pileup_generator):
                self.bam = bam
                self.pileup_generator = pileup_generator

                self.labels = []
                #TODO: Load labels and training data

            def __len__(self):
                # TODO: Get length from loaded dataset
                return 1000

            def __getitem__(self, idx):
                # TODO: Get chrom, pos, ref, alt, var_type labels from dataset
                chrom = 1
                pos = 10
                var_type = 2
                pileup = self.pileup_generator(self.bam, chrom, pos)
                return pileup, var_type

        self.dataloader = DataLoader(DatasetWrapper(bam, labels, pileup_generator),
                                     batch_size = batch_size, shuffle = shuffle,
                                     num_workers = num_workers)


    def __len__(self):
        return len(self.dataloader)

    @property
    def data_iterator(self):
        return self.dataloader

    @property
    def dataset(self):
        return None
