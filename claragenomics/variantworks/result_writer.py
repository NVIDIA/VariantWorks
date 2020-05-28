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

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import mkdtemp
import vcf
from vcf.parser import _Format, _Substitution, _Call
from vcf.model import make_calldata_tuple

from claragenomics.variantworks.types import VariantZygosity


class ResultWriter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def write_output(self):
        pass


class VCFResultWriter(ResultWriter):

    zygosity_to_vcf_genotype = {
        VariantZygosity.NO_VARIANT:     "0/0",
        VariantZygosity.HETEROZYGOUS:   "0/1",
        VariantZygosity.HOMOZYGOUS:     "1/1",
    }

    def __init__(self, variant_label_loader, infered_zygosities, output_location=None):
        self.vcf_path_to_reader_writer = dict()
        self.variant_label_loader = variant_label_loader
        self.infered_zygosities = infered_zygosities
        self.output_location = Path(output_location) if output_location else Path(mkdtemp())

    def _get_encoded_zygosity_to_genotype(self, idx):
        return VCFResultWriter.zygosity_to_vcf_genotype[self.infered_zygosities[idx]]

    def _get_formatted_vcf_record_for_variant(self, idx, variant):
        output_line = \
            [variant.chrom, variant.pos, variant.id, variant.ref, variant.allele,
             variant.quality, variant.filter,
             variant.info + ';IZ=' + self._get_encoded_zygosity_to_genotype(idx),
             variant.format]
        output_line = [str(entry) if entry is not None else '.' for entry in output_line]
        # We don't support multisample - only set  the value for the first sample since
        output_line += variant.samples
        return output_line

    @staticmethod
    def _get_modified_reader_headers(vcf_file_path, append_to_format_headers):
        vcf_reader = vcf.Reader(filename=str(vcf_file_path))
        last_format_header_line_index = \
            max([line_number for line_number, hline in enumerate(vcf_reader._header_lines) if "INFO=" in hline])
        new_headers = \
            vcf_reader._header_lines[0:last_format_header_line_index + 1] + \
            append_to_format_headers + \
            vcf_reader._header_lines[last_format_header_line_index + 1:]
        new_headers.append('#' + '\t'.join(vcf_reader._column_headers + vcf_reader.samples))
        return '\n'.join(new_headers) + '\n'

    def _get_variant_file_writer(self, variant):
        vcf_file_path = Path(variant.vcf)
        if vcf_file_path not in self.vcf_path_to_reader_writer:
            vcf_writer = open(self.output_location / (vcf_file_path.name + '.vcf'), 'w')
            vcf_writer.write(self._get_modified_reader_headers(
                vcf_file_path, ['##INFO=<ID=IZ,Number=1,Type=String,Description="Inferred Zygosity Results">']
            ))
            self.vcf_path_to_reader_writer[vcf_file_path] = vcf_writer
        return self.vcf_path_to_reader_writer[vcf_file_path]

    def write_output(self):
        # Iterate over all variances
        for idx, variant in enumerate(self.variant_label_loader):
            file_writer = self._get_variant_file_writer(variant)
            file_writer.write(
                '\t'.join(self._get_formatted_vcf_record_for_variant(idx, variant)) + '\n')
        # Close all file writers
        for _, fwriter in self.vcf_path_to_reader_writer.items():
            fwriter.close()
