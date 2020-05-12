# Abstract class for creating a dataset from BAM and VCF files

from torch.utils.data import Dataset, DataLoader
import vcf

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import *

from claragenomics.variantworks.base_encoder import base_enum_encoder
from claragenomics.variantworks.neural_types import VariantEncodingType, VariantAlleleType, VariantZygosityType
from claragenomics.variantworks.types import VariantZygosity

class VariantDataLoader(DataLayerNM):
    """
    Data layer that outputs (variant type, variant allele, variant position) tuples.

    Args:
        variant_encoder : Encoder for variant input
        label_loader : Label loader object
        label_encoder : An encoder for labels
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
            "vz_label": NeuralType(tuple('B'), VariantZygosityType()),
            "encoding": NeuralType(('B', 'C', 'H', 'W'), VariantEncodingType()),
        }

    def __init__(self, variant_encoder, label_loader, label_encoder, batch_size=32, shuffle=True, num_workers=4):
        super().__init__()

        class DatasetWrapper(Dataset):
            def __init__(self, variant_encoder, label_loader, label_encoder):
                self.label_loader = label_loader
                self.label_encoder = label_encoder
                self.variant_encoder = variant_encoder

            def __len__(self):
                return len(self.label_loader)

            def __getitem__(self, idx):
                variant = self.label_loader[idx]
                #print(variant)

                encoding = self.variant_encoder(variant)
                target = self.label_encoder(variant)
                return target, encoding

        self.dataloader = DataLoader(DatasetWrapper(variant_encoder, label_loader, label_encoder),
                                     batch_size = batch_size, shuffle = shuffle,
                                     num_workers = num_workers)
        self.variant_encoder = variant_encoder

    def __len__(self):
        return len(self.dataloader)

    @property
    def data_iterator(self):
        return self.dataloader

    @property
    def dataset(self):
        return None
