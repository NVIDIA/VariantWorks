#
# Copyright 2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Classes and functions for encoding samples."""

import abc
from datetime import datetime
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import pysam
import torch

from variantworks.types import FileRegion, Variant, VariantZygosity
from variantworks.utils.visualization import rgb_to_hex
from variantworks.utils.encoders import find_insertions, normalize_counts, calculate_positions, reencode_base_pileup


# Torch multiprocessing limits interferes with python mp module. Using this helps resolve
# error as described in https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935/2
torch.multiprocessing.set_sharing_strategy('file_system')


class Encoder:
    """An abstract class defining the interface to an encoder implementation.

    Encoder could be used for encoding inputs to network, as well as encoding target labels for prediction.
    """

    def __init__(self):
        """Construct a class instance."""
        pass

    @abc.abstractmethod
    def __call__(self, *sample):
        """Compute the encoding of a sample."""
        raise NotImplementedError


class SummaryEncoder(Encoder):
    """A summary count encoder for pileups.

    For a given pileup of reads (e.g. output from samtools mpileup), the encoder generates
    tensor for each pileup column. The encoder counts the number of DNA bases (A, G, G, T, deletion)
    for each pileup column on both the forward and reverse strands. Insertions are handled by encoding
    new pileup columns. Therefore, the output of the encoder is a tensor of shape (num_pileup_col, 10).
    The output of this encoder can be used to train a sequence aware model such such as an RNN.

    This encoding is inspired by a featurizer used in Medaka
    (https://github.com/nanoporetech/medaka/blob/master/medaka/features.py)
    """

    def __init__(self, exclude_no_coverage_positions=True, normalize_counts=True, use_quality=False):
        """Constructor for the class.

        Args:
            exclude_no_coverage_positions : Flag to determine if pileup columns with 0
                                            coverage should be dropped.
            normalize_counts : Flag to determine if summary counts in encoding should
                               be normalized.
            use_quality : Flag to indicate if draft base and quality is to be encoded.

        Returns:
            Instance of class.
        """
        self._exclude_no_coverage_positions = exclude_no_coverage_positions
        self._normalize_counts = normalize_counts
        self._use_quality = use_quality

        # Supported alphabet when building summary encoder.
        self.symbols = ["a",
                        "c",
                        "g",
                        "t",
                        "A",
                        "C",
                        "G",
                        "T",
                        "#",
                        "*"]
        self.draft_symbols = ["A",
                              "C",
                              "G",
                              "T",
                              "*"]

    def __call__(self, region, ref_quality=None):
        """Generate a torch tensor with summary encoding.

        Args:
            region : Region dataclass specifying region within a pileup to generate
                     an encoding for.
            ref_quality : A list with base quality values of draft sequence.

        Returns:
            Tuple of (count_matrix, positions)
            count_matrix : A torch tensor encoding the summary count for the pileup.
            If quality score is enabled, rows with draft base and draft base
            quality are encoded in count matrix as well.
            positions : A torch tensor encoding reference and inserted positions in pileup
        """
        assert(isinstance(region, FileRegion))
        assert(not self._use_quality or ref_quality is not None),\
            "Encoder initialized to use quality but not quality scores passes."
        start_pos = region.start_pos
        end_pos = region.end_pos
        pileup_file = region.file_path

        # Load pileup file into a dataframe
        pileup = pd.read_csv(pileup_file, delimiter="\t", header=None, quoting=3).values

        if (end_pos is None):
            end_pos = len(pileup)

        if (len(pileup) < end_pos):
            end_pos = len(pileup)

        subreads = pileup[:, 4]
        truth_coverage = pileup[:, 7].astype("int")
        positions = calculate_positions(start_pos, end_pos, subreads, truth_coverage,
                                        self._exclude_no_coverage_positions)

        positions = torch.IntTensor(positions)
        # Using positions, calculate pileup counts
        num_features = len(self.symbols)
        if ref_quality:
            num_features += len(self.draft_symbols) + 1  # Extra 1 for the base quality
        pileup_counts = torch.zeros((len(positions), num_features))
        for i in range(len(positions)):
            ref_position = positions[i][0]
            insert_position = positions[i][1]
            base_pileup = subreads[ref_position].strip("^]").strip("$")
            base_pileup = reencode_base_pileup(pileup[ref_position, 2], base_pileup)
            insertions, next_to_del = find_insertions(base_pileup)
            insertions_to_keep = []

            # Remove all insertions which are next to delete positions in pileup
            for k in range(len(insertions)):
                if next_to_del[k] is False:
                    insertions_to_keep.append(insertions[k])

            # Replace all occurrences of insertions from the pileup string
            for insertion in insertions:
                base_pileup = base_pileup.replace("+" + str(len(insertion)) + insertion, "")

            if (insert_position == 0):  # No insertions for this position
                for j in range(len(self.symbols)):
                    pileup_counts[i, j] = base_pileup.count(self.symbols[j])
                # Add draft base and base quality to encoding
                if ref_quality:
                    pileup_counts[i, len(self.symbols) + self.draft_symbols.index(pileup[ref_position, 2])] = 1
                    pileup_counts[i, len(self.draft_symbols) + len(self.symbols)] = ref_quality[ref_position] / 93.0
            elif (insert_position > 0):
                # Remove all insertions which are smaller than minor position being considered
                # so we only count inserted bases at positions longer than the minor position
                insertions_minor = [x for x in insertions_to_keep if len(x) >= insert_position]
                for j in range(len(insertions_minor)):
                    inserted_base = insertions_minor[j][insert_position-1]
                    pileup_counts[i, self.symbols.index(inserted_base)] += 1

        if self._normalize_counts:
            return normalize_counts(pileup_counts, positions), positions
        else:
            return pileup_counts, positions


class HaploidLabelEncoder(Encoder):
    """A haploid label encoder for pileups containing truth sequence.

    Given a pileup generated from a truth sequence aligned to the draft sequence,
    generate one-hot encoded labels for each pileup column. The possible labels
    should be one of: [A, C, G, T, deletion], in both the forward and reverse strand.
    Therefore, given a pileup file as input, the output labels should be of shape
    (num_pileup_col, 10).

    This encoding is inspired by a label encoder used in Medaka
    (https://github.com/nanoporetech/medaka/blob/master/medaka/labels.py)
    """

    def __init__(self, exclude_no_coverage_positions=True):
        """Constructor for the class.

        Returns:
            Instance of class.
        """
        self._exclude_no_coverage_positions = exclude_no_coverage_positions

        # Supported alphabet when building haploid label encoder.
        self.symbols = ["*",
                        "A",
                        "C",
                        "G",
                        "T"]

    def __call__(self, region):
        """Generate a torch tensor with summary encoding.

        Args:
            region : Region dataclass specifying region within a pileup to generate
                     an encoding for.

        Returns:
            (labels, positions) tuple
            labels : A torch tensor encoding the labels for the pileup
            positions : A torch tensor encoding reference and inserted positions in labels
        """
        assert(isinstance(region, FileRegion))
        start_pos = region.start_pos
        end_pos = region.end_pos
        pileup_file = region.file_path

        # Load pileup file into a dataframe
        pileup = pd.read_csv(pileup_file, delimiter="\t", header=None, quoting=3).values

        if (end_pos is None):
            end_pos = len(pileup)

        if (len(pileup) < end_pos):
            end_pos = len(pileup)

        subreads = pileup[:, 4]
        truth_coverage = pileup[:, 7].astype("int")
        positions = calculate_positions(start_pos, end_pos, subreads, truth_coverage,
                                        self._exclude_no_coverage_positions)

        positions = torch.IntTensor(positions)
        # Using positions, calculate pileup counts
        truth = pileup[:, 8]
        labels = np.zeros((len(positions),))  # gap, A, C, G, T (sparse format)
        for i in range(len(positions)):
            reference_pos = positions[i][0]
            inserted_pos = positions[i][1]
            truth_base = truth[reference_pos].strip("^]").strip("$").upper()
            truth_base = reencode_base_pileup(pileup[reference_pos, 2], truth_base)
            # Handle minor position label (no insertion)
            if (inserted_pos == 0):
                if ("+" in truth_base):
                    ref_base = truth_base.split("+")[0]
                    labels[i] = self.symbols.index(ref_base)
                elif ("-" in truth_base):
                    ref_base = truth_base.split("-")[0]
                    labels[i] = self.symbols.index(ref_base)
                elif (truth_base in self.symbols):
                    labels[i] = self.symbols.index(truth_base)
            # Handle major position label (with insertion)
            elif (inserted_pos > 0):
                if ("+" in truth_base):
                    inserted_bases = truth_base.split("+")[1]
                    inserted_bases = "".join(
                        [i for i in inserted_bases if i.isdigit() is False])
                    if (len(inserted_bases) >= inserted_pos):
                        inserted_truth_base = inserted_bases[inserted_pos - 1].upper()
                        labels[i] = self.symbols.index(inserted_truth_base)
            else:
                raise RuntimeError(
                    "Encode labels error - inserted position should be >= 0.")
        return torch.from_numpy(labels), positions


class BaseEnumEncoder(Encoder):
    """An Enum encoder that returns an output encoding for Nucleotide base.

    Converts Nucleotide base char type to a class number.
    """

    def __init__(self):
        """Construct a class instance."""
        super().__init__()
        self._dict = {
            'A': 1,
            'a': 1,
            'T': 2,
            't': 2,
            'C': 3,
            'c': 3,
            'G': 4,
            'g': 4,
            'N': 5,
            'n': 5,
        }

    def __call__(self, nucleotide):
        """Encode Nucleotide base to Enum.

        Returns:
           Nucleotide base encoded as number.
        """
        assert(nucleotide in self._dict)
        return self._dict[nucleotide]


class BaseUnicodeEncoder(Encoder):
    """A Unicode code encoder that returns an output encoding for Nucleotide base.

    Converts Nucleotide base char type to a Unicode numeric value.
    """

    def __init__(self):
        """Construct a class instance."""
        super().__init__()
        self._nucleotides = ['A', 'a', 'T', 't', 'C', 'c', 'G', 'g', 'N', 'n']

    def __call__(self, nucleotide):
        """Encode Nucleotide base to Unicode code.

        Returns:
           Nucleotide base encoded as Unicode code.
        """
        assert(nucleotide in self._nucleotides)
        return ord(nucleotide)


class UnicodeRGBEncoder(Encoder):
    """A encoder that returns an RGB color encoding for Nucleotide base Unicode value.

    Converts Nucleotide base unicode value type to a RGB color list.
    """

    def __init__(self):
        """Construct a class instance."""
        super().__init__()
        self._dict = {
            ord('\0'):  [255, 255, 255],    # white (null char for cells initiated to 'zero' )
            ord('A'):   [0, 128, 0],        # green
            ord('a'):   [0, 128, 0],        # green
            ord('T'):   [255, 0, 0],        # red
            ord('t'):   [255, 0, 0],        # red
            ord('C'):   [0, 0, 255],        # blue
            ord('c'):   [0, 0, 255],        # blue
            ord('G'):   [255, 255, 0],      # yellow
            ord('g'):   [255, 255, 0],      # yellow
            ord('N'):   [0, 0, 0],          # black
            ord('n'):   [0, 0, 0]           # black
        }

    def __call__(self, nucleotide_unicode):
        """Encode Nucleotide base to Unicode code.

        Returns:
           Nucleotide base encoded as Unicode code.
        """
        assert(nucleotide_unicode in self._dict)
        return self._dict[nucleotide_unicode]

    def get_keys(self):
        """Get nucleotide bases unicode keys."""
        return set(x for x in self._dict.keys() if chr(x) != '\0' and chr(x).upper() == chr(x))

    @staticmethod
    def get_key_legend_label(k):
        """Get keys corresponding name for legend ."""
        return chr(k)


class PileupEncoder(Encoder):
    """A pileup encoder for BAMs.

    For a given SNP position and nucleotide context, the encoder generates a pileup
    tensor around the variant position. The pileup can have configurable depth based on
    the type of information that is selected to be embedded.

    The variant location of interest is kept centered in the pileup, and the layers input in
    the constructor define the channels created in the encoding. For more details on available
    channels, please check the documentation for the Layers enum.
    """

    class Layer(Enum):
        r"""Layers that can be added to the pileup encoding.

        Values:
            READ : Encode each aligned read as a row of the pileup. The bases in the
            read are encoded using a base_encoder dict passed into the class. The reads
            in the row are positioned according to the pileup alignment.

            BASE_QUALITY : Encode the base quality of each aligned read in the pileup. Base
            qualities of each read are added to a new row, following the same positioning as for READS. The base
            qualities are normalized to [0,1] (using max value of 93 per SAM format).
            Missing base quality is set to 0.

            MAPPING_QUALITY : Mapping quality of a read is encoded at each nucleotide position of the read. Mapping
            quality values are noramlize to [0,1] (assuming max value of 50).
            Missing mapping quality is set to 0.

            REFERENCE : Only the reference allele location is encoded in each row.

            ALLELE : Only the alt allele location is encoded in each row.
        """

        READ = 0
        BASE_QUALITY = 1
        MAPPING_QUALITY = 2
        REFERENCE = 3
        ALLELE = 4

    def __init__(self, window_size=50, max_reads=50, layers=[Layer.READ], base_encoder=None, print_encoding=False):
        """Construct class instance.

        Args:
            window_size : A nucleotide context size on either side of variant position [50].
            max_reads : Max number of reads to consider in the pileip. If reads fewer than max_reads
            are available, the entries are all masked to 0. [50]
            layers : A list defining the layers to add to the encoding. The ordering of channels in the
            encoding follows the ordering of layers in the list. [Layer.READ]
            base_encoder : A class which inherits from `Encoder` defining conversion of nucleotide string chars to
            numeric representation in its __call__ method. [BaseEnumEncoder]
            print_encoding : Print ASCII representation of each encoding that's converted to a tensor. [False]

        Returns:
            Instance of class.
        """
        super().__init__()
        self.window_size = window_size
        self.max_reads = max_reads
        self.layers = layers
        self.bams = dict()
        self.base_encoder = base_encoder if base_encoder is not None else BaseEnumEncoder()
        self.layer_tensors = []
        self.layer_dict = {}
        for layer in layers:
            tensor = torch.zeros(
                (self.height, self.width), dtype=torch.float32)
            self.layer_tensors.append(tensor)
            self.layer_dict[layer] = tensor
        self.print_encoding = print_encoding

    @property
    def width(self):
        """Return width of pileup."""
        return 2 * self.window_size + 1

    @property
    def height(self):
        """Return height of pileup."""
        return self.max_reads

    @property
    def depth(self):
        """Return number of layers in pileup."""
        return len(self.layers)

    def _fill_layer(self, layer, pileupread, left_offset, right_offset, row, pileup_pos_range, variant):
        # print(len(pileupread.alignment.get_reference_sequence()))
        tensor = self.layer_dict[layer]

        query_pos = pileupread.query_position

        # Currently only support adding reads
        if layer == self.Layer.READ:
            # Fetch the subsequence based on the offsets
            seq = pileupread.alignment.query_sequence[query_pos -
                                                      left_offset: query_pos + right_offset]
            if self.print_encoding:
                print("{}{}{}".format("-" * pileup_pos_range[0], seq, "-" *
                                      (2 * self.window_size + 1 - len(seq) - pileup_pos_range[0])))
            for seq_pos, pileup_pos in enumerate(range(pileup_pos_range[0], pileup_pos_range[1])):
                # Encode base characters to enum
                tensor[row, pileup_pos] = self.base_encoder(seq[seq_pos])
        elif layer == self.Layer.BASE_QUALITY:
            # From SAM format docs.
            MAX_BASE_QUALITY = 93.0
            # Fetch the subsequence based on the offsets
            seq_qual = pileupread.alignment.query_qualities[query_pos -
                                                            left_offset: query_pos + right_offset]
            for seq_pos, pileup_pos in enumerate(range(pileup_pos_range[0], pileup_pos_range[1])):
                # Encode base characters to enum
                qual = seq_qual[seq_pos]
                if qual == 255:
                    qual = 0.
                else:
                    qual = qual / MAX_BASE_QUALITY
                tensor[row, pileup_pos] = qual
        elif layer == self.Layer.MAPPING_QUALITY:
            MAX_MAPPING_QUALITY = 100.0
            # Getch mapping quality of alignment
            map_qual = pileupread.alignment.mapping_quality
            # Missing mapiping quality is 255
            if map_qual == 255:
                map_qual = 0.0
            else:
                map_qual = pileupread.alignment.mapping_quality / MAX_MAPPING_QUALITY
            for pileup_pos in range(pileup_pos_range[0], pileup_pos_range[1]):
                # Encode base characters to enum
                tensor[row, pileup_pos] = map_qual
        elif layer == self.Layer.REFERENCE:
            if self.print_encoding:
                print("{}{}{}".format("-" * self.window_size, variant.ref, "-" *
                                      (2 * self.window_size + 1 - len(variant.ref) - self.window_size)))
            # Only encode the reference at the variant position, rest all 0
            for seq_pos, pileup_pos in enumerate(
                    range(self.window_size, min(self.window_size + len(variant.ref), 2 * self.window_size - 1))):
                tensor[row, pileup_pos] = self.base_encoder(variant.ref[seq_pos])
        elif layer == self.Layer.ALLELE:
            if self.print_encoding:
                print("{}{}{}".format("-" * self.window_size, variant.allele, "-" *
                                      (2 * self.window_size + 1 - len(variant.allele) - self.window_size)))
            # Only encode the allele at the variant position, rest all 0
            for seq_pos, pileup_pos in enumerate(
                    range(self.window_size, min(self.window_size + len(variant.allele), 2 * self.window_size - 1))):
                tensor[row, pileup_pos] = self.base_encoder(variant.allele[seq_pos])

    def __call__(self, variant):
        """Return a torch Tensor pileup queried from a BAM file.

        Args:
            variant : Variant struct holding information about variant locus.
        """
        # This encoding supports only single sample encoding.
        if len(variant.samples) != 1:
            raise RuntimeError("{} only supports single sample VCFs.".format(self.__class__.__name__))

        # Locus information
        chrom = variant.chrom
        variant_pos = variant.pos
        bam_file = variant.bams[0]

        # Check that the ref and alt alleles all fit in the window context.
        if len(variant.ref) > self.window_size:
            raise RuntimeError("Ref allele {} too large for window {}. Please increase window size.".format(
                variant.ref, self.window_size))
        if len(variant.allele) > self.window_size:
            raise RuntimeError("Alt allele {} too large for window {}. Please increase window size.".format(
                variant.allele, self.window_size))

        # Create BAM object if one hasn't been opened before.
        if bam_file not in self.bams:
            self.bams[bam_file] = pysam.AlignmentFile(bam_file, "rb")

        bam = self.bams[bam_file]

        # Get pileups from BAM.
        # Note that VCF positions are 1 based, but pysam pileup regions are 0 based.
        # So subtract one from position.
        pileups = bam.pileup(chrom,
                             variant_pos - 1, variant_pos,
                             truncate=True,
                             max_depth=self.max_reads)

        if self.print_encoding:
            print("\nEncoding for {}".format(variant))
            print("Order of rows : {}".format(self.layers))

        for col, pileup_col in enumerate(pileups):
            for row, pileupread in enumerate(pileup_col.pileups):
                # Skip rows beyond the max depth
                if row >= self.max_reads:
                    break
                # Check if reference base is missing (either deleted or skipped).
                if pileupread.is_del or pileupread.is_refskip:
                    continue

                if pileupread.is_head or pileupread.is_tail:
                    continue

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
                right_offset = min(self.window_size + 1, len(pileupread.alignment.query_sequence) -
                                   pileupread.query_position)

                pileup_pos_range = (
                    self.window_size - left_offset, self.window_size + right_offset)
                for layer in self.layers:
                    self._fill_layer(layer, pileupread, left_offset,
                                     right_offset, row, pileup_pos_range, variant)

        encoding = torch.stack(self.layer_tensors)
        [tensor.zero_() for tensor in self.layer_tensors]
        return encoding

    def visualize(self, variant, save_to_path=None, max_subplots_per_line=3, visual_decoder=UnicodeRGBEncoder()):
        """Visualize variant encoded pileup.

        Outputs variant pileup visualization to a figure.
        Execute `tensorboard --logdir='<save_to_path>' --port=6006` in the background to view the images over
        TensorBoard.

        Args:
            variant: Variant struct holding information about variant locus.
            save_to_path: Path to figure output direcoty. [None]
            max_subplots_per_line: maximal number of plots per row in the figure. [3]
            visual_decoder: a decoder for a visualized representation of PileupEncoder.base_encoder
        Returns:
            figure_title: figure title
            figure: matplotlib.pyplot.figure object
        """

        def _get_subplots_axes():
            cols = len(self.layers) if len(self.layers) < max_subplots_per_line else max_subplots_per_line
            # Calculate the ceil() value of len(self.layers) divided by max_subplots_per_line
            rows = (len(self.layers) + (max_subplots_per_line - 1)) // max_subplots_per_line
            return rows, cols

        def _create_subplot(idx, nrow, ncol, layer, sample_dim):
            plt.subplot(nrow, ncol, idx)
            plt_name = 'Layer: {}'.format(layer.name)
            plt.title(plt_name,
                      loc='left',
                      fontdict={'fontsize': 7})
            plt.ylabel('Read number')
            plt.xlabel("Read window size")
            if layer in [PileupEncoder.Layer.READ,
                         PileupEncoder.Layer.REFERENCE,
                         PileupEncoder.Layer.ALLELE]:
                data = sample_dim.numpy().astype(np.uint8)
                rgb_img = np.zeros((sample_dim.shape[0], sample_dim.shape[1], 3))
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        rgb_img[i, j, :] = visual_decoder(data[i, j])
                plt.imshow(rgb_img)
                plt.legend(
                    handles=[
                        mpatches.Patch(
                            facecolor=rgb_to_hex(visual_decoder(nucleotide_unicode)),
                            edgecolor='black',
                            label=visual_decoder.get_key_legend_label(nucleotide_unicode)
                        )
                        for nucleotide_unicode in visual_decoder.get_keys()
                    ],
                    bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=1
                )
            if layer in [PileupEncoder.Layer.MAPPING_QUALITY, PileupEncoder.Layer.BASE_QUALITY]:
                plt.imshow(sample_dim.numpy(), cmap='Purples')
                plt.colorbar(orientation='vertical', pad=0.02)

        encoded_sample = self.__call__(variant)  # Build variant pileup encoding
        figure = plt.figure(figsize=(20, 10))
        figure_title = 'chrom-{}_pos-{}'.format(variant.chrom, variant.pos) + \
                       ('_id-{}'.format(variant.id) if variant.id != '.' else '')
        figure.suptitle(figure_title, fontweight="bold", y=1)
        # Determine the number of rows and cols in multiple subplots figure
        number_rows, number_column = _get_subplots_axes()
        for index, sample_layer, encoded_sample_layer in zip(range(1, len(self.layers)+1), self.layers, encoded_sample):
            _create_subplot(index, number_rows, number_column, sample_layer, encoded_sample_layer)
        if save_to_path is not None:
            try:
                plt.savefig(os.path.join(
                    save_to_path, figure_title + '_{}.png'.format(datetime.today().strftime('%Y-%m-%d'))
                ))
            except FileNotFoundError as e:
                raise e
        return figure_title, figure


class ZygosityLabelEncoder(Encoder):
    """A label encoder that returns an output label encoding for zygosity only.

    Converts zygosity type to a class number.
    """

    def __init__(self):
        """Construct a class instance."""
        super().__init__()
        self._dict = {
            VariantZygosity.NO_VARIANT: 0,
            VariantZygosity.HOMOZYGOUS: 1,
            VariantZygosity.HETEROZYGOUS: 2,
        }

    def __call__(self, variant):
        """Encode variant to class for zygosity.

        Returns:
           Zygosity encoded as number.
        """
        # This encoding supports only single sample encoding.
        if len(variant.samples) != 1:
            raise RuntimeError("{} only supports single sample VCFs.".format(self.__class__.__name__))

        assert(isinstance(variant, Variant))
        var_zyg = variant.zygosity[0]
        assert(var_zyg in self._dict)

        return torch.tensor(self._dict[var_zyg])


class ZygosityLabelDecoder(Encoder):
    """A decoder to convert a class to a zygosity enum."""

    def __init__(self):
        """Construct a class instance."""
        super().__init__()
        self._dict = {
            0: VariantZygosity.NO_VARIANT,
            1: VariantZygosity.HOMOZYGOUS,
            2: VariantZygosity.HETEROZYGOUS,
        }

    def __call__(self, class_id):
        """Decode class to variant zygosity enum.

        Returns:
            Variant zygosity.
        """
        assert(class_id.item() in self._dict)
        return self._dict[class_id.item()]
