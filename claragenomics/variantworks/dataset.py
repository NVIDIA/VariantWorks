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
            "va_label": NeuralType(tuple('B'), VariantAlleleType()),
            "encoding": NeuralType(axes=(
                                   AxisType(kind=AxisKind.Batch, size=None, is_list=False),
                                   AxisType(kind=AxisKind.Channel, size=self.variant_encoder.depth, is_list=False),
                                   AxisType(kind=AxisKind.Height, size=self.variant_encoder.height, is_list=False),
                                   AxisType(kind=AxisKind.Width, size=self.variant_encoder.width, is_list=False),
                                   ), elements_type=VariantEncodingType()),
        }

    def __init__(self, variant_encoder, label_loader, batch_size=32, shuffle=True, num_workers=4):
        super().__init__()

        class DatasetWrapper(Dataset):
            def __init__(self, variant_encoder, label_loader):
                self.label_loader = label_loader
                self.variant_encoder = variant_encoder

            def __len__(self):
                return len(self.label_loader)

            def __getitem__(self, idx):
                variant = self.label_loader[idx]
                var_zyg = variant.zygosity
                var_allele = variant.allele
                #print(chrom, pos, ref, var_zyg, var_allele)
                if var_zyg == VariantZygosity.NO_VARIANT:
                    var_zyg = 0
                elif var_zyg == VariantZygosity.HOMOZYGOUS:
                    var_zyg = 1
                elif var_zyg == VariantZygosity.HETEROZYGOUS:
                    var_zyg = 2

                encoding = self.variant_encoder.encode(variant)
                return var_zyg, base_enum_encoder[var_allele], encoding

        self.dataloader = DataLoader(DatasetWrapper(variant_encoder, label_loader),
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
