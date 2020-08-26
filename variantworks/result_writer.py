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
import math
import os
from pathlib import Path
import re
from tempfile import mkdtemp
import vcf
import multiprocessing as mp

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
    """A writer that outputs contents of a VCFReader to an output VCF."""

    zygosity_to_vcf_genotype = {
        VariantZygosity.NONE:           "./.",
        VariantZygosity.NO_VARIANT:     "0/0",
        VariantZygosity.HETEROZYGOUS:   "0/1",
        VariantZygosity.HOMOZYGOUS:     "1/1",
    }

    def __init__(self, variant_loader, output_location=None, num_threads=mp.cpu_count()):
        """Construct a VCFResultWriter class.

        Args:
            variant_loader : An instance of a VCF loader class.
            output_location : Output directory for result VCFs.
            num_threads : Number of threads to use for writing VCF output.

        Returns:
            Instance of class.
        """
        self.variant_loader = variant_loader
        self.output_location = Path(
            output_location) if output_location else Path(mkdtemp())
        self._num_threads = num_threads

    @staticmethod
    def _serialize_record_info(info_dict):
        ret_list = list()
        for k, v in info_dict.items():
            if v is None:
                continue
            if not isinstance(v, str) and math.isnan(v):
                continue
            if type(v) is list:
                ret_list.append("{}={}".format(
                    k, ','.join(map(lambda x: "{:.4f}".format(x) if isinstance(x, float) else str(x), v))))
            elif isinstance(v, bool):
                if v:
                    ret_list.append(str(k))
            else:
                ret_list.append("{}={:.4f}".format(k, v) if isinstance(v, float) else "{}={}".format(k, v))
        return ";".join(ret_list) if ret_list else "."

    @staticmethod
    def _serialize_record_sample(sample):
        ret_list = list()
        for field_value in sample:
            if not field_value or type(field_value) is None:
                ret_list.append(".")
            elif type(field_value) is list:
                ret_list.append(",".join(["{:.4f}".format(x) if isinstance(x, float) else str(x) for x in field_value]))
            else:
                ret_list.append(str(field_value))
        return ":".join(ret_list) if ret_list else "."

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

    def _get_serialized_vcf_record_for_variant(self, idx):
        variant = self.variant_loader[idx]
        output_line = \
            [variant.chrom, variant.pos + 1, variant.id, variant.ref,
             variant.allele, variant.quality, self._serialize_record_filter(
                 variant.filter),
             self._serialize_record_info(variant.info), ':'.join(variant.format)]
        output_line = [
            str(entry) if entry is not None else '.' for entry in output_line]
        variant.samples = [self._serialize_record_sample(
            sample) for sample in variant.samples]
        output_line += variant.samples
        return output_line

    @staticmethod
    def _get_original_headers_from_vcf_reader(file_path):
        reader = vcf.Reader(filename=str(file_path))
        return reader._header_lines, reader._column_headers, reader.samples

    def _get_modified_reader_headers(self, vcf_file_path, append_to_format_headers):
        vcf_headers, vcf_column_headers, vcf_reader_samples_name =\
            VCFResultWriter._get_original_headers_from_vcf_reader(vcf_file_path)
        modified_headers_metadata = vcf_headers
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
            '#' + '\t'.join(vcf_column_headers + self.variant_loader.samples))
        return '\n'.join(modified_headers_metadata) + '\n'

    def _init_file_writer(self, vcf_file_path):
        if vcf_file_path:
            vcf_base_name = os.path.basename(vcf_file_path)
            vcf_base_name = vcf_base_name.split('.')[0]
            if not os.path.exists(self.output_location):
                os.makedirs(self.output_location)
            vcf_writer = open(os.path.join(
                self.output_location, '{}_{}.{}'.format("inferred", vcf_base_name, 'vcf')), 'w')
            vcf_writer.write(
                self._get_modified_reader_headers(vcf_file_path, []))
            return vcf_writer
        else:
            raise RuntimeError("No VCF file path found in variant entry.")

    def write_output(self):
        """Write final output to file."""
        file_writer = self._init_file_writer(self.variant_loader[0].vcf)
        pool = mp.Pool(self._num_threads)
        # Iterate over all variances
        for line in pool.imap(self._get_serialized_vcf_record_for_variant,
                              range(len(self.variant_loader)),
                              chunksize=50000):
            file_writer.write('\t'.join(line) + '\n')
        # Close all file writers
        file_writer.close()
