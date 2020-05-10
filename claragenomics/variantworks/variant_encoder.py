# Classes and functions to encode pileups

import abc
import pysam
import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import NeuralType, ChannelType
from nemo.utils.decorators import add_port_docs
from nemo.core.neural_factory import DeviceType

from claragenomics.variantworks.base_encoder import base_enum_encoder
from claragenomics.variantworks.neural_types import VariantPositionType

class BaseEncoder(NonTrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports
        """
        return {
            "variant_pos": NeuralType(tuple('B'), VariantPositionType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports
        """
        return {
            "pileup": NeuralType(('B', 'C', 'H', 'W'), ChannelType()),
        }

    def __init__(self):
        super().__init__()

    def forward(self, variant_pos):
        """Generates a batch of variant encodings.
        This function is required to be implemented in NonTrainableNM inherited classes.

        Args:
            variant_pos : Batch of variant positions.

        Returns:
            torch.tensor of stacked variant encodings.
        """
        examples = zip(*variant_pos)
        #print("===================")
        #for e in examples:
        #    print(e)
        #print("===================")
        tensors = [self.encode(example[0], example[1], example[2]) for example in examples]
        pileup = torch.stack(tensors)
        device = torch.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        pileup = pileup.to(device)
        return pileup

    @abc.abstractmethod
    def encode(self, bam_file, chrom, variant_pos):
        """Return an encoding of a variant position.
        """

class SnpPileupEncoder(BaseEncoder):
    """A pileup encoder for SNVs. For a given SNP position and base context, the encoder
    generates a pileup tensor around the variant position.
    """
    def __init__(self, window_size = 50, max_reads = 50, channels={"reads"}):
        super().__init__()
        self.window_size = window_size
        self.max_reads = max_reads
        self.channels = channels
        self.bams = dict()

    @property
    def size(self):
        return (len(self.channels), self.max_reads, 2 * self.window_size + 1)

    def encode(self, bam_file, chrom, variant_pos):
        """Returns a torch Tensor pileup queried from a BAM file.

        Args:
            bam_file : Path to bam file
            chrom : String for chromosome in BAM file
            variant_pos : Locus of variant in BAM
        """
        # Create BAM object if one hasn't been opened before.
        if (bam_file not in self.bams):
            self.bams[bam_file] = pysam.AlignmentFile(bam_file, "rb")

        bam = self.bams[bam_file]

        encoding = torch.zeros(self.size, dtype=torch.float32)

        # Get pileups from BAM
        pileups = bam.pileup(chrom,
                             variant_pos, variant_pos + 1,
                             truncate=True,
                             max_depth = self.max_reads)

        for col, pileup_col in enumerate(pileups):
            for row, pileupread in enumerate(pileup_col.pileups):
                # Skip rows beyond the max depth
                if row >= self.max_reads:
                    break
                # Check of reference base is missing (either deleted or skipped).
                assert(not pileupread.is_del and not pileupread.is_refskip)

                # Position of variant locus in read
                query_pos = pileupread.query_position

                # Using the variant locus as the center, find the left and right offset
                # from that locus to use as bounds for fetching bases from reads.
                #
                #      |------V------|
                #  ATCGATCGATCGATCG
                #        ATCGATCGATCGATCGATCG
                #
                # 1st read - Left offset is window size, and right offset is 4 bases
                # 2nd read - Left offset is 5 bases, and right offset is window size
                left_offset = min(self.window_size, pileupread.query_position)
                right_offset = min(self.window_size + 1, len(pileupread.alignment.query_sequence) - pileupread.query_position)

                # Fetch the subsequence based on the offsets
                sub_seq = pileupread.alignment.query_sequence[query_pos - left_offset: query_pos + right_offset]

                # Currently only support adding reads
                if "reads" in self.channels:
                    for seq_pos, pileup_pos in enumerate(range(self.window_size - left_offset, self.window_size + right_offset)):
                        # Encode base characters to enum
                        encoding[0, row, pileup_pos] = base_enum_encoder[sub_seq[seq_pos]]

        return encoding
