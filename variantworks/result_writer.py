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
"""Set of functions and classes to write results of inference to various output formats."""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import re
from tempfile import mkdtemp
import vcf

from variantworks.types import VariantZygosity


class ResultWriter(ABC):
    """Abstract base class for result writers."""

    @abstractmethod
    def __init__(self):
        """Construct a ResultWriter."""
        pass

    @abstractmethod
    def write_output(self):
        """Output results."""
        pass


class VCFResultWriter(ResultWriter):
    """A result writer that outputs predicted zygosities to VCFs."""

    zygosity_to_vcf_genotype = {
        VariantZygosity.NO_VARIANT:     "0/0",
        VariantZygosity.HETEROZYGOUS:   "0/1",
        VariantZygosity.HOMOZYGOUS:     "1/1",
    }

    def __init__(self, variant_label_loader, inferred_zygosities=None, output_location=None):
        """Construct a VCFResultWriter class.

        Args:
            variant_label_loader : An instance of a VCF loader class.
            inferred_zygosities : A list of inferred zygosity for each variant in VCF loader.
            output_location : Output directory for result VCFs.

        Returns:
            Instance of class.
        """
        # Check if the inferred zygosities passed in are the same length as label loader
        if inferred_zygosities and (len(inferred_zygosities) != len(variant_label_loader)):
            raise RuntimeError(
                "VCFResultWriter needs an inferred zygosity per entry in the sample loader.")
        self.vcf_path_to_reader_writer = dict()
        self.variant_label_loader = variant_label_loader
        self.inferred_zygosities = inferred_zygosities
        self.output_location = Path(
            output_location) if output_location else Path(mkdtemp())

    def _update_format(self, variant, idx):
        # We don't support multisample - only set the inferred GT value for the first sample
        try:
            gt_format_index = variant.format.index('GT')
        except ValueError:
            variant.format.append('GT')
            gt_format_index = 0
        variant.samples[0][gt_format_index] = VCFResultWriter.zygosity_to_vcf_genotype[self.inferred_zygosities[idx]]

    @staticmethod
    def _serialize_record_info(info_dict):
        ret_list = list()
        for k, v in info_dict.items():
            if type(v) is list:
                ret_list.append("{}={}".format(
                    k, ','.join(map(lambda x: str(x), v))))
            elif type(v) is bool:
                ret_list.append(str(k))
            else:
                ret_list.append("{}={}".format(k, str(v)))
        return ";".join(ret_list)

    @staticmethod
    def _serialize_record_sample(sample):
        ret_list = list()
        for field_value in sample:
            if not field_value or type(field_value) is None:
                ret_list.append(".")
            elif type(field_value) is list:
                ret_list.append(",".join([str(x) for x in field_value]))
            else:
                ret_list.append(str(field_value))
        return ":".join(ret_list)

    @staticmethod
    def _serialize_record_filter(var_filter):
        if var_filter is None:
            return "."
        elif type(var_filter) is list:
            if len(var_filter) == 0:
                return "PASS"
            else:
                return ";".join(var_filter)
        else:
            return str(var_filter)

    def _get_serialized_vcf_record_for_variant(self, idx, variant):
        output_line = \
            [variant.chrom, variant.pos, variant.id, variant.ref,
             variant.allele, variant.quality, self._serialize_record_filter(
                 variant.filter),
             self._serialize_record_info(variant.info), ':'.join(variant.format)]
        output_line = [
            str(entry) if entry is not None else '.' for entry in output_line]
        # We don't support multisample - only set the inferred GT value for the first sample
        self._update_format(variant, idx)
        variant.samples = [self._serialize_record_sample(
            sample) for sample in variant.samples]
        output_line += variant.samples
        return output_line

    @staticmethod
    def _get_modified_reader_headers(vcf_file_path, append_to_format_headers):
        vcf_reader = vcf.Reader(filename=str(vcf_file_path))
        modified_headers_metadata = vcf_reader._header_lines
        for meta_data_line in append_to_format_headers:
            metadata_type_to_search = re.search(
                '##(.*=<)', meta_data_line).group(1)
            last_format_header_line_index = \
                max([line_number for line_number, hline in enumerate(modified_headers_metadata)
                     if metadata_type_to_search in hline])
            modified_headers_metadata = \
                modified_headers_metadata[0:last_format_header_line_index + 1] + \
                [meta_data_line] + \
                modified_headers_metadata[last_format_header_line_index + 1:]
        modified_headers_metadata.append(
            '#' + '\t'.join(vcf_reader._column_headers + vcf_reader.samples))
        return '\n'.join(modified_headers_metadata) + '\n'

    def _get_variant_file_writer(self, variant):
        vcf_file_path = os.path.abspath(variant.vcf)
        if vcf_file_path not in self.vcf_path_to_reader_writer:
            vcf_base_name = os.path.basename(vcf_file_path)
            vcf_base_name = vcf_base_name.split('.')[0]
            if not os.path.exists(self.output_location):
                os.makedirs(self.output_location)
            vcf_writer = open(os.path.join(
                self.output_location, '{}_{}.{}'.format("inferred", vcf_base_name, 'vcf')), 'w')
            vcf_writer.write(
                self._get_modified_reader_headers(vcf_file_path, []))
            self.vcf_path_to_reader_writer[vcf_file_path] = vcf_writer
        return self.vcf_path_to_reader_writer[vcf_file_path]

    def write_output(self):
        """Write final output to file."""
        # Iterate over all variances
        for idx, variant in enumerate(self.variant_label_loader):
            file_writer = self._get_variant_file_writer(variant)
            file_writer.write(
                '\t'.join(self._get_serialized_vcf_record_for_variant(idx, variant)) + '\n')
        # Close all file writers
        for _, fwriter in self.vcf_path_to_reader_writer.items():
            fwriter.close()
