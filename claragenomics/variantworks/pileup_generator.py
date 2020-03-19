# Classes and functions to encode pileups

import pysam
import torch

from claragenomics.variantworks.base_encoder import base_enum_encoder

class SnpPileupGenerator:
    def __init__(self, window_size = 50, max_reads = 50, channels={"reads"}):
        self.window_size = window_size
        self.max_reads = max_reads
        self.channels = channels
        self.bams = dict()

    @property
    def size(self):
        return (len(self.channels), self.max_reads, 2 * self.window_size + 1)

    def __call__(self, bam_file, chrom, variant_pos):
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
                # TODO: Only there for SNP pileup
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
