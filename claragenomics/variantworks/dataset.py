# Abstract class for creating a dataset from BAM and VCF files

from torch.utils.data import Dataset, DataLoader
import vcf

from nemo.backends.pytorch.nm import DataLayerNM
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_types import ChannelType, LabelsType, LossType, NeuralType

from claragenomics.variantworks.base_encoder import base_enum_encoder

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
            "vt_label": NeuralType(tuple('B'), LabelsType()),
            "va_label": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(self, bam, labels, pileup_generator, batch_size=32, shuffle=True, num_workers=4):
        super().__init__()

        class DatasetWrapper(Dataset):
            def __init__(self, bam, labels, pileup_generator):
                self.bam = bam
                self.pileup_generator = pileup_generator

                self.labels = self.parse_vcf_labels(labels)
                #TODO: Load labels and training data

            def __len__(self):
                # TODO: Get length from loaded dataset
                return len(self.labels)

            def __getitem__(self, idx):
                # TODO: Get chrom, pos, ref, alt, var_type labels from dataset
                chrom, pos, ref, var_type, var_allele, var_all_seq = self.labels[idx]
                pileup = self.pileup_generator(self.bam, chrom, pos)
                #print(chrom, pos, ref, var_type, var_allele, var_all_seq)
                return pileup, var_type, var_allele

            def parse_vcf_labels(self, vcf_file):
                labels = []
                assert(vcf_file[-3:] == ".gz") # Check for compressed file
                vcf_reader = vcf.Reader(open(vcf_file, "rb"))
                for record in vcf_reader:
                    assert(record.is_snp) # Right now only supporting SNPs
                    chrom = record.CHROM
                    pos = record.POS
                    ref = record.REF
                    var_allele = base_enum_encoder[record.ALT[0].sequence]
                    var_type = 0 # None
                    if record.num_het > 0:
                        var_type = 2 # Heterozygous
                    elif record.num_hom_alt > 0:
                        var_type = 1 # Homozygous

                    labels.append((chrom, pos, ref, var_type, var_allele, record.ALT[0].sequence))
                return labels

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
