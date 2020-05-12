# Classes and functions to encode pileups

import abc
from enum import Enum
import pysam
import torch

from claragenomics.variantworks.base_encoder import base_enum_encoder
from claragenomics.variantworks.types import Variant, VariantType, VariantZygosity

class BaseEncoder():
    """An abstract class defining the interface to an encoder implementation. Encoder could
    be used for encoding inputs to network, as well as encoding target labels for prediction.
    """
    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def size(self):
        """Return size of encoding.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, variant):
        """Computes the encoding of a variant location.
        """
        raise NotImplementedError


class PileupEncoder(BaseEncoder):
    """A pileup encoder for SNVs. For a given SNP position and base context, the encoder
    generates a pileup tensor around the variant position.
    """
    class Layer(Enum):
        """Layers that can be added to the pileup encoding.
        """
        READ = 0
        BASE_QUALITY = 1
        MAPPING_QUALITY = 2

    def __init__(self, window_size = 50, max_reads = 50, layers=[Layer.READ]):
        super().__init__()
        self.window_size = window_size
        self.max_reads = max_reads
        self.layers = layers
        self.bams = dict()
        self.layer_tensors = []
        self.layer_dict = {}
        for layer in layers:
            tensor = torch.zeros((self.height, self.width), dtype=torch.float32)
            self.layer_tensors.append(tensor)
            self.layer_dict[layer] = tensor

    @property
    def width(self):
        return 2 * self.window_size + 1

    @property
    def height(self):
        return self.max_reads

    @property
    def depth(self):
        return len(self.layers)

    @property
    def size(self):
        return ((self.depth, self.height, self.width))

    def _fill_layer(self, layer, pileupread, left_offset, right_offset, row, pileup_pos_range):
        """Generate encoding for requested layer in pileup.
        """
        tensor = self.layer_dict[layer]

        query_pos = pileupread.query_position

        # Currently only support adding reads
        if layer == self.Layer.READ:
            # Fetch the subsequence based on the offsets
            seq = pileupread.alignment.query_sequence[query_pos - left_offset: query_pos + right_offset]
            for seq_pos, pileup_pos in enumerate(range(pileup_pos_range[0], pileup_pos_range[1])):
                # Encode base characters to enum
                tensor[row, pileup_pos] = base_enum_encoder[seq[seq_pos]]
        elif layer == self.Layer.BASE_QUALITY:
            # Fetch the subsequence based on the offsets
            seq_qual = pileupread.alignment.query_qualities[query_pos - left_offset: query_pos + right_offset]
            for seq_pos, pileup_pos in enumerate(range(pileup_pos_range[0], pileup_pos_range[1])):
                # Encode base characters to enum
                tensor[row, pileup_pos] = seq_qual[seq_pos]
        elif layer == self.Layer.MAPPING_QUALITY:
             # Getch mapping quality of alignment
            map_qual = pileupread.alignment.mapping_quality
            for pileup_pos in range(pileup_pos_range[0], pileup_pos_range[1]):
                # Encode base characters to enum
                tensor[row, pileup_pos] = map_qual


    def __call__(self, variant):
        """Returns a torch Tensor pileup queried from a BAM file.

        Args:
            bam_file : Path to bam file
            variant : Variant struct holding information about variant locus
        """
        # Locus information
        chrom = variant.chrom
        variant_pos = variant.pos
        bam_file = variant.bam

        assert(variant.type == VariantType.SNP), "Only SNP variants supported in PileupEncoder currently."

        # Create BAM object if one hasn't been opened before.
        if (bam_file not in self.bams):
            self.bams[bam_file] = pysam.AlignmentFile(bam_file, "rb")

        bam = self.bams[bam_file]

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

                pileup_pos_range = (self.window_size - left_offset, self.window_size + right_offset)
                for layer in self.layers:
                    self._fill_layer(layer, pileupread, left_offset, right_offset, row, pileup_pos_range)

        encoding = torch.stack(self.layer_tensors)
        [tensor.zero_() for tensor in self.layer_tensors]
        return encoding

class ZygosityLabelEncoder(BaseEncoder):
    """A label encoder that returns an output label encoding for zygosity
    only. Converts zygosity type to a class number.
    """
    def __init__(self):
        super().__init__()
        pass

    def size(self):
        return ((1))

    def __call__(self, variant):
        assert(isinstance(variant, Variant))
        var_zyg = variant.zygosity
        if var_zyg == VariantZygosity.NO_VARIANT:
            var_zyg = 0
        elif var_zyg == VariantZygosity.HOMOZYGOUS:
            var_zyg = 1
        elif var_zyg == VariantZygosity.HETEROZYGOUS:
            var_zyg = 2

        return torch.tensor(var_zyg)
