# Abstract class for creating a dataset from BAM and VCF files

from torch.utils.data import Dataset, DataLoader
import vcf

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import ChannelType, LabelsType, LossType, NeuralType

from claragenomics.variantworks.base_encoder import base_enum_encoder
from claragenomics.variantworks.neural_types import VariantPositionType, VariantAlleleType, VariantType
from claragenomics.variantworks.types import VariantZygosity

class VariantDataLoader(DataLayerNM):
    """
    Data layer that outputs (variant type, variant allele, variant position) tuples.

    Args:
        bam : Path to BAM file
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
            "vt_label": NeuralType(tuple('B'), VariantType()),
            "va_label": NeuralType(tuple('B'), VariantAlleleType()),
            "variant_pos": NeuralType(tuple('B'), VariantPositionType()),
        }

    def __init__(self, bam, label_loader, batch_size=32, shuffle=True, num_workers=4):
        super().__init__()

        class DatasetWrapper(Dataset):
            def __init__(self, bam, label_loader):
                self.bam = bam

                self.label_loader = label_loader

            def __len__(self):
                return len(self.label_loader)

            def __getitem__(self, idx):
                chrom, pos, ref, var_type, var_allele = self.label_loader[idx]
                #print(chrom, pos, ref, var_type, var_allele)
                if var_type == VariantZygosity.NO_VARIANT:
                    var_type = 0
                elif var_type == VariantZygosity.HOMOZYGOUS:
                    var_type = 1
                elif var_type == VariantZygosity.HETEROZYGOUS:
                    var_type = 2
                return int(var_type), base_enum_encoder[var_allele], (self.bam, chrom, pos)

        self.dataloader = DataLoader(DatasetWrapper(bam, label_loader),
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
